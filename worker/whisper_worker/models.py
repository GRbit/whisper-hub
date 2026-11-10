"""Model lifecycle and job scheduling.

Each model runs one job at a time (an asyncio lock per model, so waiting
jobs cost no thread). Models load lazily, are evicted least recently used
when WHISPER_HUB_ENGINE_MAX_LOADED_MODELS is reached, and unload after
WHISPER_HUB_ENGINE_MODEL_IDLE_TIMEOUT seconds without use.
"""

import asyncio
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
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.handle: Any = None
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
        # True while loading, so concurrent loads of other models count it against the cap.
        self.reserved = False

    @property
    def loaded(self) -> bool:
        return self.handle is not None


class Job:
    """Job is one queued or running piece of work; events arrive on queue."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancel_event = threading.Event()
        self.task: Optional[asyncio.Task] = None

    def cancel(self) -> None:
        self.cancel_event.set()


class ModelManager:
    def __init__(self, cfg: Config, engine: Engine):
        self.cfg = cfg
        self.engine = engine
        self.slots: Dict[str, Slot] = {}
        self._cond: Optional[asyncio.Condition] = None
        self._reaper: Optional[asyncio.Task] = None
        # One thread per model is enough (one job per model) plus room for loads/unloads.
        self.executor = ThreadPoolExecutor(max_workers=len(cfg.models) + 2, thread_name_prefix="engine")

    async def start(self) -> None:
        self._cond = asyncio.Condition()
        self.slots = {m: Slot(m) for m in self.cfg.models}
        if self.cfg.model_idle_timeout > 0:
            self._reaper = asyncio.create_task(self._reap_idle())
        if self.cfg.preload:
            slot = self.slots[self.cfg.default_model]
            async with slot.lock:
                await self._ensure_loaded(slot)

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

    def submit(self, model_id: str, work: Callable[[Any, Job], Any], label: str) -> Job:
        """submit queues work(handle, job) on a thread; its outcome ends up on job.queue.

        Events: whatever work emits via job.queue, then ("result", value) or ("error", WorkerError).
        """
        job = Job(model_id)
        job.task = asyncio.create_task(self._run(job, work, label))
        return job

    def emitter(self, job: Job, loop: asyncio.AbstractEventLoop) -> Callable[[dict], None]:
        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(job.queue.put_nowait, ("event", event))

        return emit

    async def _run(self, job: Job, work, label: str) -> None:
        slot = self.slots[job.model_id]
        queued = time.monotonic()
        if slot.lock.locked():
            logger.debug("%s queued: model %s busy", label, slot.model_id)
        try:
            async with slot.lock:
                if job.cancel_event.is_set():
                    logger.debug("%s dropped: client left while queued", label)
                    job.queue.put_nowait(("error", Unavailable("cancelled")))
                    return
                logger.debug("%s started after %.3fs in queue", label, time.monotonic() - queued)
                await self._ensure_loaded(slot)
                slot.last_used = time.monotonic()
                try:
                    loop = asyncio.get_running_loop()
                    value = await loop.run_in_executor(self.executor, work, slot.handle, job)
                    job.queue.put_nowait(("result", value))
                except JobCancelled:
                    logger.debug("%s stopped: client disconnected", label)
                    job.queue.put_nowait(("error", Unavailable("cancelled")))
                finally:
                    slot.last_used = time.monotonic()
        except WorkerError as e:
            job.queue.put_nowait(("error", e))
        except Exception:
            logger.exception("%s failed", label)
            job.queue.put_nowait(("error", Internal("transcription failed")))
        finally:
            async with self._cond:
                self._cond.notify_all()

    async def _ensure_loaded(self, slot: Slot) -> None:
        """_ensure_loaded loads slot's model; the caller holds slot.lock."""
        if slot.loaded:
            return
        loop = asyncio.get_running_loop()
        async with self._cond:
            while True:
                used = [s for s in self.slots.values() if (s.loaded or s.reserved) and s is not slot]
                if len(used) < self.cfg.max_loaded_models:
                    break
                idle = sorted((s for s in used if s.loaded and not s.lock.locked()), key=lambda s: s.last_used)
                if idle:
                    victim = idle[0]
                    # Uncontended asyncio.Lock.acquire does not yield, so nobody can grab it in between.
                    await victim.lock.acquire()
                    try:
                        logger.info("Unloading model %s to make room for %s", victim.model_id, slot.model_id)
                        await self._unload(victim)
                    finally:
                        victim.lock.release()
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
                if slot.loaded and not slot.lock.locked() and now - slot.last_used > timeout:
                    await slot.lock.acquire()
                    try:
                        if slot.loaded:
                            logger.info("Unloading model %s after %ds idle", slot.model_id, timeout)
                            await self._unload(slot)
                    finally:
                        slot.lock.release()
                    async with self._cond:
                        self._cond.notify_all()
