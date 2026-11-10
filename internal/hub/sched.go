package hub

import (
	"cmp"
	"context"
	"slices"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Scheduling policy
//
// Every worker runs at most MaxInFlight jobs. Jobs that cannot start wait in
// one hub-wide list bounded by QueueLength. Whenever a slot frees or a
// worker appears, waiting jobs are considered in order of priority
// (highest first), then arrival (FIFO), and each goes to the least loaded
// healthy worker serving its model that has a free slot. A job whose model
// has no free worker does not block jobs for other models.
//
// Waiting does not raise a job's priority: a constant stream of
// higher-priority jobs keeps lower ones waiting. Running jobs are never
// preempted.
//
// When the queue is full, a new job evicts the newest waiting job of
// strictly lower priority (which fails with KindUnavailable); if there is
// none, the new job itself is rejected with KindUnavailable.

type job struct {
	model string
	prio  core.Priority
	seq   uint64
	enq   time.Time
	ready chan struct{} // closed once w or err is set
	w     *worker
	err   error
}

// acquire waits for a slot on a worker serving model. The caller must
// release the returned worker. queued is the time spent waiting.
func (h *Hub) acquire(ctx context.Context, model string, prio core.Priority) (w *worker, queued time.Duration, err error) {
	prio = max(core.PriorityBatch, min(prio, core.PriorityInteractive))
	h.mu.Lock()
	if !h.servedLocked(model) {
		h.mu.Unlock()
		return nil, 0, core.Errorf(core.KindUnavailable, "model %q is not available right now", model)
	}
	h.seq++
	j := &job{model: model, prio: prio, seq: h.seq, enq: h.now(), ready: make(chan struct{})}
	h.pending = append(h.pending, j)
	h.dispatchLocked()
	if j.w == nil && j.err == nil && len(h.pending) > h.cfg.QueueLength {
		h.evictLocked(j)
	}
	h.mu.Unlock()

	select {
	case <-j.ready:
	case <-ctx.Done():
		h.mu.Lock()
		got := j.w
		if got == nil && j.err == nil {
			h.removeLocked(j)
			h.log.Debug("job canceled while queued", "model", model, "seq", j.seq)
		}
		h.mu.Unlock()
		// Assigned concurrently with the cancel: honor the cancel but give
		// the slot back.
		if got != nil {
			h.release(got)
		}
		return nil, h.now().Sub(j.enq), core.WrapError(core.KindCanceled, ctx.Err(), "request canceled")
	}
	queued = h.now().Sub(j.enq)
	if j.err != nil {
		return nil, queued, j.err
	}
	return j.w, queued, nil
}

// release frees a slot and starts the next eligible job.
func (h *Hub) release(w *worker) {
	h.mu.Lock()
	w.inFlight--
	h.dispatchLocked()
	h.mu.Unlock()
}

func (h *Hub) removeLocked(j *job) {
	if i := slices.Index(h.pending, j); i >= 0 {
		h.pending = slices.Delete(h.pending, i, i+1)
	}
}

func (h *Hub) failLocked(j *job, err error) {
	h.removeLocked(j)
	j.err = err
	close(j.ready)
}

// evictLocked makes room for j, which is already in pending.
func (h *Hub) evictLocked(j *job) {
	var victim *job
	for _, p := range h.pending {
		if p == j {
			continue
		}
		if victim == nil || p.prio < victim.prio || (p.prio == victim.prio && p.seq > victim.seq) {
			victim = p
		}
	}
	full := core.Errorf(core.KindUnavailable, "server busy, queue is full")
	if victim != nil && victim.prio < j.prio {
		h.log.Debug("queue full, evicting lower priority job", "evicted_seq", victim.seq, "for_seq", j.seq)
		h.failLocked(victim, full)
		return
	}
	h.log.Debug("queue full, rejecting job", "seq", j.seq, "queued", len(h.pending)-1)
	h.failLocked(j, full)
}

// dispatchLocked starts waiting jobs on free workers in policy order.
func (h *Hub) dispatchLocked() {
	if len(h.pending) == 0 {
		return
	}
	now := h.now()
	slices.SortStableFunc(h.pending, func(a, b *job) int {
		if a.prio != b.prio {
			return int(b.prio) - int(a.prio)
		}
		return cmp.Compare(a.seq, b.seq)
	})
	kept := h.pending[:0]
	for _, j := range h.pending {
		if w := h.pickLocked(j.model); w != nil {
			w.inFlight++
			j.w = w
			close(j.ready)
			h.log.Debug("job dispatched", "model", j.model, "seq", j.seq, "priority", j.prio,
				"worker", w.url, "in_flight", w.inFlight, "waited", now.Sub(j.enq))
			continue
		}
		kept = append(kept, j)
	}
	clear(h.pending[len(kept):])
	h.pending = kept
}

// pickLocked returns the least loaded healthy worker for model with a free
// slot, rotating between equally loaded ones; nil when none is free.
func (h *Hub) pickLocked(model string) *worker {
	var best *worker
	bestLoad := 2.0
	n := len(h.workers)
	for i := range n {
		w := h.workers[(h.rr+i)%n]
		if w.inFlight >= w.limit || !w.serves(model) {
			continue
		}
		if load := float64(w.inFlight) / float64(w.limit); load < bestLoad {
			best, bestLoad = w, load
		}
	}
	if best != nil {
		h.rr++
	}
	return best
}

// failUnservableLocked fails waiting jobs whose model lost its last worker;
// they would otherwise wait for an unknown time.
func (h *Hub) failUnservableLocked() {
	for _, j := range slices.Clone(h.pending) {
		if !h.servedLocked(j.model) {
			h.log.Debug("failing queued job, model gone", "model", j.model, "seq", j.seq)
			h.failLocked(j, core.Errorf(core.KindUnavailable, "model %q is not available right now", j.model))
		}
	}
}
