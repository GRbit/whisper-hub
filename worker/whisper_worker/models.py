"""Model lifecycle and job scheduling.

Each model runs one job at a time (a slot per model with an asyncio wait
queue, so waiting jobs cost no thread). A running job may pause at a pause
point while a pause request names it (POST /v1/pause): it gives the model to
the next job and takes it back, ahead of other waiters, once no pause
request remains. Models load lazily, are evicted least recently used when
WHISPER_HUB_ENGINE_MAX_LOADED_MODELS is reached, and unload after
WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT seconds without use. A model with a
paused job is never evicted: the job still holds engine state.
"""

import asyncio
import collections
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .engines.base import Engine
from .errors import Internal, JobCancelled, ModelNotFound, Unavailable, WorkerError

logger = logging.getLogger("whisper_worker")


class Slot:
    """Slot is one model: its handle and who may use it right now."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.handle: Any = None
        self.last_used = time.monotonic()
        # True while loading, so concurrent loads of other models count it against the cap.
        self.reserved = False
        self.holder: Any = None
        self.waiters: collections.deque = collections.deque()
        self.paused: set = set()

    @property
    def loaded(self) -> bool:
        return self.handle is not None

    def busy(self) -> bool:
        return self.holder is not None or bool(self.paused)

    async def acquire(self, owner, front: bool = False) -> None:
        if self.holder is None and not self.waiters:
            self.holder = owner
            return
        fut = asyncio.get_running_loop().create_future()
        entry = (owner, fut)
        if front:
            self.waiters.appendleft(entry)
        else:
            self.waiters.append(entry)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                try:
                    self.waiters.remove(entry)
                except ValueError:
                    pass
            raise

    def try_acquire(self, owner) -> bool:
        if self.holder is None and not self.waiters:
            self.holder = owner
            return True
        return False

    def release(self) -> None:
        self.holder = None
        while self.waiters:
            owner, fut = self.waiters.popleft()
            if not fut.done():
                self.holder = owner
                fut.set_result(None)
                return


class Job:
    """Job is one queued or running piece of work; events arrive on queue."""

    def __init__(self, model_id: str, job_id: Optional[str] = None, pausable: bool = False):
        self.model_id = model_id
        self.job_id = job_id
        self.pausable = pausable
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancel_event = threading.Event()
        # Set while at least one pause request names this job.
        self.pause_event = threading.Event()
        # The engine thread waits on it while paused.
        self.wake = threading.Event()
        self.task: Optional[asyncio.Task] = None
        self.finished = asyncio.Event()
        self._changed = asyncio.Event()
        self.on_pause_point: Optional[Callable[[], None]] = None

    def cancel(self) -> None:
        self.cancel_event.set()
        self.wake.set()
        self.notify()

    def notify(self) -> None:
        """notify wakes coroutines waiting for a pause or cancel change; loop thread only."""
        self._changed.set()
        self._changed = asyncio.Event()

    async def changed(self) -> None:
        await self._changed.wait()

    def checkpoint(self) -> None:
        """checkpoint runs on the engine thread at a pause point and blocks while paused."""
        if not self.pausable or not self.pause_event.is_set() or self.cancel_event.is_set():
            return
        self.wake.clear()
        self.on_pause_point()
        self.wake.wait()


class ModelManager:
    def __init__(self, cfg: Config, engine: Engine):
        self.cfg = cfg
        self.engine = engine
        self.slots: Dict[str, Slot] = {}
        self._cond: Optional[asyncio.Condition] = None
        self._reaper: Optional[asyncio.Task] = None
        # Loads and unloads only; jobs run on threads of their own, since a
        # paused job keeps its thread and nested pauses would exhaust a pool.
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="engine-load")
        self.holds: Dict[str, int] = {}
        self.by_id: Dict[str, Job] = {}
        self._registered = asyncio.Event()

    async def start(self) -> None:
        self._cond = asyncio.Condition()
        self.slots = {m: Slot(m) for m in self.cfg.models}
        if self.cfg.model_idle_timeout > 0:
            self._reaper = asyncio.create_task(self._reap_idle())
        if self.cfg.preload:
            slot = self.slots[self.cfg.default_model]
            await slot.acquire("preload")
            try:
                await self._ensure_loaded(slot)
            finally:
                slot.release()

    async def stop(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def resolve(self, model: Optional[str]) -> str:
        if not model:
            return self.cfg.default_model
        if model not in self.slots:
            raise ModelNotFound(f"model {model!r} is not served by this worker")
        return model

    def describe(self) -> List[dict]:
        return [
            {"id": m, "default": m == self.cfg.default_model, "loaded": self.slots[m].loaded}
            for m in self.cfg.models
        ]

    def submit(self, model_id: str, work: Callable[[Any, Job], Any], label: str, job_id: Optional[str] = None,
               pausable: bool = False) -> Job:
        """submit queues work(handle, job) on a thread; its outcome ends up on job.queue.

        Events: whatever work emits via job.queue (including paused and resumed),
        then ("result", value) or ("error", WorkerError).
        """
        job = Job(model_id, job_id, pausable)
        loop = asyncio.get_running_loop()
        job.on_pause_point = lambda: loop.call_soon_threadsafe(self._paused, job, label)
        if job_id:
            self.by_id[job_id] = job
            if self.holds.get(job_id, 0) > 0:
                job.pause_event.set()
            self._registered.set()
            self._registered = asyncio.Event()
        job.task = asyncio.create_task(self._run(job, work, label))
        return job

    def emitter(self, job: Job, loop: asyncio.AbstractEventLoop) -> Callable[[dict], None]:
        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(job.queue.put_nowait, ("event", event))

        return emit

    # --- pause requests ---

    def pause_fits(self, job_id: str, model: Optional[str]) -> bool:
        """pause_fits tells whether model can run while job_id stays paused with its model loaded."""
        job = self.by_id.get(job_id)
        if job is None or not model or model == job.model_id or model not in self.slots:
            return True
        if self.slots[model].loaded:
            return True
        own = self.slots[job.model_id]
        kept = [s for s in self.slots.values() if (s.loaded or s.reserved) and (s.busy() or s is own)]
        return len(kept) < self.cfg.max_loaded_models

    def hold(self, job_id: str, delta: int) -> None:
        n = self.holds.get(job_id, 0) + delta
        if n > 0:
            self.holds[job_id] = n
        else:
            self.holds.pop(job_id, None)
        job = self.by_id.get(job_id)
        if job is not None:
            if n > 0:
                job.pause_event.set()
            else:
                job.pause_event.clear()
            job.notify()

    async def wait_ended(self, job_id: str) -> None:
        """wait_ended returns once the job named job_id has arrived and ended."""
        while True:
            job = self.by_id.get(job_id)
            if job is not None:
                await job.finished.wait()
                return
            await self._registered.wait()

    def _paused(self, job: Job, label: str) -> None:
        """_paused runs on the loop when job's engine thread reached a pause point."""
        if job.cancel_event.is_set() or not job.pause_event.is_set():
            job.wake.set()
            return
        slot = self.slots[job.model_id]
        slot.paused.add(job)
        slot.release()
        job.queue.put_nowait(("event", {"type": "paused"}))
        logger.debug("%s paused, model %s lent to the next job", label, job.model_id)
        asyncio.create_task(self._resume(job, slot, label))

    async def _resume(self, job: Job, slot: Slot, label: str) -> None:
        started = time.monotonic()
        try:
            while job.pause_event.is_set() and not job.cancel_event.is_set():
                await job.changed()
            if job.cancel_event.is_set():
                return
            take = asyncio.ensure_future(slot.acquire(job, front=True))
            while not take.done() and not job.cancel_event.is_set():
                change = asyncio.ensure_future(job.changed())
                await asyncio.wait({take, change}, return_when=asyncio.FIRST_COMPLETED)
                change.cancel()
            if not take.done():
                take.cancel()
                return
            if job.cancel_event.is_set():
                slot.release()
                return
            job.queue.put_nowait(("event", {"type": "resumed"}))
            logger.debug("%s resumed after %.3fs", label, time.monotonic() - started)
        finally:
            slot.paused.discard(job)
            job.wake.set()

    # --- running ---

    async def _run(self, job: Job, work, label: str) -> None:
        slot = self.slots[job.model_id]
        queued = time.monotonic()
        if slot.busy():
            logger.debug("%s queued: model %s busy", label, slot.model_id)
        try:
            await slot.acquire(job)
            try:
                if job.cancel_event.is_set():
                    logger.debug("%s dropped: client left while queued", label)
                    job.queue.put_nowait(("error", Unavailable("cancelled")))
                    return
                logger.debug("%s started after %.3fs in queue", label, time.monotonic() - queued)
                await self._ensure_loaded(slot)
                slot.last_used = time.monotonic()
                try:
                    value = await _in_thread(work, slot.handle, job)
                    job.queue.put_nowait(("result", value))
                except JobCancelled:
                    logger.debug("%s stopped: client disconnected", label)
                    job.queue.put_nowait(("error", Unavailable("cancelled")))
                finally:
                    slot.last_used = time.monotonic()
            finally:
                # A job canceled while paused ends without the model.
                if slot.holder is job:
                    slot.release()
        except WorkerError as e:
            job.queue.put_nowait(("error", e))
        except Exception:
            logger.exception("%s failed", label)
            job.queue.put_nowait(("error", Internal("transcription failed")))
        finally:
            slot.paused.discard(job)
            job.finished.set()
            if job.job_id and self.by_id.get(job.job_id) is job:
                del self.by_id[job.job_id]
            async with self._cond:
                self._cond.notify_all()

    async def _ensure_loaded(self, slot: Slot) -> None:
        """_ensure_loaded loads slot's model; the caller holds slot."""
        if slot.loaded:
            return
        loop = asyncio.get_running_loop()
        async with self._cond:
            while True:
                used = [s for s in self.slots.values() if (s.loaded or s.reserved) and s is not slot]
                if len(used) < self.cfg.max_loaded_models:
                    break
                idle = sorted((s for s in used if s.loaded and not s.busy()), key=lambda s: s.last_used)
                if idle and idle[0].try_acquire("evict"):
                    victim = idle[0]
                    try:
                        logger.info("Unloading model %s to make room for %s", victim.model_id, slot.model_id)
                        await self._unload(victim)
                    finally:
                        victim.release()
                    continue
                logger.debug("Model %s waits for a free model slot", slot.model_id)
                await self._cond.wait()
            slot.reserved = True
        try:
            started = time.monotonic()
            logger.info("Loading model %s (engine %s)", slot.model_id, self.engine.name)
            try:
                slot.handle = await loop.run_in_executor(self.executor, self.engine.load, slot.model_id)
            except Exception:
                logger.exception("Loading model %s failed", slot.model_id)
                raise Unavailable("model could not be loaded") from None
            logger.info("Loaded model %s in %.1fs", slot.model_id, time.monotonic() - started)
        finally:
            slot.reserved = False

    async def _unload(self, slot: Slot) -> None:
        handle, slot.handle = slot.handle, None
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self.engine.unload, handle)
        except Exception:
            logger.exception("Unloading model %s failed", slot.model_id)

    async def _reap_idle(self) -> None:
        timeout = self.cfg.model_idle_timeout
        # Upstream polls every 15 s; shorter timeouts get a proportionally faster poll.
        interval = min(15.0, max(0.5, timeout / 4))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for slot in list(self.slots.values()):
                if slot.loaded and not slot.busy() and now - slot.last_used > timeout and slot.try_acquire("reap"):
                    try:
                        logger.info("Unloading model %s after %ds idle", slot.model_id, timeout)
                        await self._unload(slot)
                    finally:
                        slot.release()
                    async with self._cond:
                        self._cond.notify_all()


async def _in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def settle(value, exc):
        if fut.done():
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(value)

    def run():
        try:
            value = fn(*args)
        except BaseException as e:  # noqa: B902  re-raised on the loop
            loop.call_soon_threadsafe(settle, None, e)
            return
        loop.call_soon_threadsafe(settle, value, None)

    threading.Thread(target=run, name="engine-job", daemon=True).start()
    return await fut
