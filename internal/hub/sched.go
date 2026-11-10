package hub

import (
	"cmp"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Scheduling policy
//
// Every worker runs at most MaxInFlight jobs. Jobs that cannot start wait in
// one hub-wide list bounded by QueueLength. Whenever a slot frees or a
// worker appears, waiting jobs are considered in order of priority
// (highest first), then arrival, and each goes to the least loaded healthy
// worker serving its model that has a free slot. A job whose model has no
// free worker does not block jobs for other models. Waiting does not raise
// a job's priority.
//
// Preemption: when a waiting job finds every slot it could use taken and
// one runs a job of strictly lower priority, the lowest such job is asked
// to pause (POST /v1/pause). Once its stream reports it paused it gives up
// its slot but stays bound to its worker, where its state lives, and
// competes for a slot there again by priority and arrival; when it gets
// one the pause request is closed and it continues where it stopped. If
// it has not paused within PreemptTimeout (or the worker refuses), it is
// canceled and queued again to restart from the beginning. Two kinds of
// job are never canceled for a restart, only paused: one that has handed
// segments to its caller (the caller would see them twice), and one that
// was restarted MaxRestarts times already (so repeated preemption cannot
// discard its work forever). If such a job does not pause, the higher job
// waits.
//
// When the queue is full, a new job evicts the newest waiting new job of
// strictly lower priority (which fails with KindUnavailable); if there is
// none, the new job itself is rejected. Jobs that already ran (paused, or
// waiting to restart) are neither counted nor evicted.

type jobState int

const (
	jobWaiting jobState = iota // in Hub.pending
	jobRunning                 // holds a slot of w
	jobPaused                  // paused on w, holds no slot
	jobIdle                    // between attempts, or finished
)

// errRestart is the cancel cause of an attempt canceled to make room.
var errRestart = errors.New("canceled to make room for a higher priority job")

type job struct {
	ctx      context.Context // the caller's
	model    string
	prio     int
	seq      uint64
	pausable bool // transcriptions; language detection has no pause point

	state    jobState
	w        *worker
	ready    chan struct{} // closed once dispatched (w set) or failed (err set)
	err      error
	since    time.Time     // start of the current wait or pause
	queued   time.Duration // total time waiting or paused
	requeued bool          // waiting again after a restart

	// The current attempt on w, set at dispatch.
	attempt   int
	id        string
	actx      context.Context
	cancel    context.CancelCauseFunc
	delivered bool // a segment reached the caller
	restarts  int
	// maxRestarts is Config.MaxRestarts when the job was created.
	maxRestarts int
	doomed      bool // canceled to be restarted
	noPause     bool // the worker refused to pause this attempt

	// Preemption: pauseFor is set on a running job being paused for a
	// waiting one, preempting on that waiting one.
	pauseFor   *job
	preempting *job
	pauseStart time.Time
	timer      *time.Timer
	unhold     context.CancelFunc // ends the pause request
}

// protected jobs are never canceled for a restart.
func (j *job) protected() bool { return j.delivered || j.restarts >= j.maxRestarts }

func (j *job) canPause() bool { return j.pausable && !j.noPause }

func (j *job) interruptible() bool { return j.canPause() || !j.protected() }

func newIDPrefix() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// schedule runs call on a worker slot by j's priority until it returns
// without having been canceled to make room; such a cancel queues j again
// and restarts call from the beginning. onPause is for the worker stream.
func (h *Hub) schedule(j *job, call func(ctx context.Context, w *worker, id string, onPause func(bool)) error) error {
	for {
		if err := h.acquire(j); err != nil {
			return err
		}
		h.mu.Lock()
		actx, w, id, attempt := j.actx, j.w, j.id, j.attempt
		h.mu.Unlock()
		err := call(actx, w, id, func(paused bool) { h.pauseEvent(j, attempt, paused) })
		if !h.endAttempt(j, err) {
			return err
		}
	}
}

// acquire waits until j is dispatched to a worker slot.
func (h *Hub) acquire(j *job) error {
	h.mu.Lock()
	if !h.servedLocked(j.model) {
		h.mu.Unlock()
		return core.Errorf(core.KindUnavailable, "model %q is not available right now", j.model)
	}
	if j.seq == 0 {
		h.seq++
		j.seq = h.seq
	}
	j.state, j.w, j.err = jobWaiting, nil, nil
	j.ready = make(chan struct{})
	j.since = h.now()
	h.pending = append(h.pending, j)
	h.dispatchLocked()
	if j.state == jobWaiting && !j.requeued && h.freshLocked() > h.cfg.QueueLength {
		h.evictLocked(j)
	}
	h.mu.Unlock()

	select {
	case <-j.ready:
		return j.err
	case <-j.ctx.Done():
	}
	h.mu.Lock()
	switch j.state {
	case jobWaiting:
		h.removeLocked(j)
		j.state = jobIdle
		j.queued += h.now().Sub(j.since)
		h.log.Debug("job canceled while queued", "model", j.model, "seq", j.seq)
		h.dispatchLocked()
	case jobRunning:
		// Dispatched concurrently with the cancel: honor the cancel but
		// give the slot back.
		h.mu.Unlock()
		h.endAttempt(j, j.ctx.Err())
		return core.WrapError(core.KindCanceled, j.ctx.Err(), "request canceled")
	}
	h.mu.Unlock()
	if j.err != nil {
		return j.err
	}
	return core.WrapError(core.KindCanceled, j.ctx.Err(), "request canceled")
}

// endAttempt frees what the attempt held and reports whether j must run
// again because the scheduler canceled it.
func (h *Hub) endAttempt(j *job, err error) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	now := h.now()
	h.dropPauseLocked(j)
	h.withdrawLocked(j)
	switch j.state {
	case jobRunning:
		j.w.inFlight--
	case jobPaused:
		j.queued += now.Sub(j.since)
	}
	if i := slices.Index(h.active, j); i >= 0 {
		h.active = slices.Delete(h.active, i, i+1)
	}
	j.state = jobIdle
	j.cancel(nil)
	restart := j.doomed && err != nil && j.ctx.Err() == nil
	if restart {
		j.restarts++
		j.requeued = true
		j.doomed, j.noPause, j.delivered = false, false, false
		h.log.Debug("job restarting", "seq", j.seq, "priority", j.prio, "model", j.model, "restarts", j.restarts)
	} else if j.doomed && err == nil {
		h.log.Debug("job finished before its cancel for restart took effect", "seq", j.seq)
	}
	h.dispatchLocked()
	return restart
}

// delivering reports whether a segment of j may go to its caller, and
// marks j as having delivered output. A job being canceled for a restart
// must not deliver anything.
func (h *Hub) delivering(j *job) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if j.doomed {
		return false
	}
	j.delivered = true
	return true
}

func (h *Hub) freshLocked() int {
	n := 0
	for _, p := range h.pending {
		if !p.requeued {
			n++
		}
	}
	return n
}

func (h *Hub) removeLocked(j *job) {
	if i := slices.Index(h.pending, j); i >= 0 {
		h.pending = slices.Delete(h.pending, i, i+1)
	}
	h.withdrawLocked(j)
}

func (h *Hub) failLocked(j *job, err error) {
	h.removeLocked(j)
	j.state = jobIdle
	j.queued += h.now().Sub(j.since)
	j.err = err
	close(j.ready)
}

// evictLocked makes room for j, which is already in pending.
func (h *Hub) evictLocked(j *job) {
	var victim *job
	for _, p := range h.pending {
		if p == j || p.requeued {
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
	h.log.Debug("queue full, rejecting job", "seq", j.seq, "queued", h.freshLocked()-1)
	h.failLocked(j, full)
}

func byPolicy(a, b *job) int {
	if a.prio != b.prio {
		return cmp.Compare(b.prio, a.prio)
	}
	return cmp.Compare(a.seq, b.seq)
}

// dispatchLocked hands free slots to waiting and paused jobs in policy
// order, then starts preemption for those still without one.
func (h *Hub) dispatchLocked() {
	cands := slices.Clone(h.pending)
	for _, j := range h.active {
		if j.state == jobPaused {
			cands = append(cands, j)
		}
	}
	if len(cands) == 0 {
		return
	}
	slices.SortStableFunc(cands, byPolicy)
	now := h.now()
	for _, j := range cands {
		switch j.state {
		case jobWaiting:
			if w := h.pickLocked(j.model); w != nil {
				h.startLocked(j, w, now)
			}
		case jobPaused:
			if j.w.inFlight < j.w.limit {
				h.resumeLocked(j, now)
			}
		}
	}
	h.pending = slices.DeleteFunc(h.pending, func(j *job) bool { return j.state != jobWaiting })
	if h.cfg.PreemptTimeout < 0 {
		return
	}
	for _, j := range cands {
		if (j.state == jobWaiting || j.state == jobPaused) && j.preempting == nil {
			if v := h.victimLocked(j); v != nil {
				h.pauseLocked(v, j, now)
			}
		}
	}
}

func (h *Hub) startLocked(j *job, w *worker, now time.Time) {
	w.inFlight++
	j.w, j.state = w, jobRunning
	j.queued += now.Sub(j.since)
	j.attempt++
	j.id = fmt.Sprintf("%s-%d-%d", h.idPrefix, j.seq, j.attempt)
	j.actx, j.cancel = context.WithCancelCause(j.ctx)
	h.active = append(h.active, j)
	h.withdrawLocked(j)
	close(j.ready)
	h.log.Debug("job dispatched", "model", j.model, "seq", j.seq, "priority", j.prio, "attempt", j.attempt,
		"worker", w.url, "in_flight", w.inFlight, "waited", now.Sub(j.since))
}

func (h *Hub) resumeLocked(j *job, now time.Time) {
	j.w.inFlight++
	j.state = jobRunning
	j.queued += now.Sub(j.since)
	if j.unhold != nil {
		j.unhold()
		j.unhold = nil
	}
	h.withdrawLocked(j)
	h.log.Debug("job resumed", "seq", j.seq, "priority", j.prio, "worker", j.w.url, "paused", now.Sub(j.since))
}

// victimLocked picks the running job to interrupt for waiting job x: the
// lowest priority below x's (the newest among equals) on a slot x could
// use, skipping jobs already being paused or that cannot be interrupted.
func (h *Hub) victimLocked(x *job) *job {
	var best *job
	for _, v := range h.active {
		if v.state != jobRunning || v.pauseFor != nil || v.doomed || v.prio >= x.prio || !v.interruptible() {
			continue
		}
		if x.state == jobPaused {
			if v.w != x.w {
				continue
			}
		} else if !v.w.serves(x.model) {
			continue
		}
		if best == nil || v.prio < best.prio || (v.prio == best.prio && v.seq > best.seq) {
			best = v
		}
	}
	return best
}

func (h *Hub) pauseLocked(v, x *job, now time.Time) {
	v.pauseFor, x.preempting, v.pauseStart = x, v, now
	h.log.Debug("asking job to pause", "seq", v.seq, "priority", v.prio, "for_seq", x.seq, "for_priority", x.prio,
		"worker", v.w.url, "pausable", v.canPause(), "protected", v.protected())
	attempt := v.attempt
	if v.canPause() {
		pctx, cancel := context.WithCancel(context.Background())
		v.unhold = cancel
		client, id, model := v.w.client, v.id, x.model
		go func() {
			if err := client.Pause(pctx, id, model); err != nil && pctx.Err() == nil {
				h.pauseRefused(v, attempt, err)
			}
		}()
	}
	if !v.protected() {
		v.timer = time.AfterFunc(h.cfg.PreemptTimeout, func() { h.pauseTimedOut(v, attempt) })
	}
}

// pauseEvent handles the paused and resumed lines of v's stream.
func (h *Hub) pauseEvent(v *job, attempt int, paused bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if v.attempt != attempt {
		return
	}
	if !paused {
		h.log.Debug("worker resumed job", "seq", v.seq)
		return
	}
	x := v.pauseFor
	if v.state != jobRunning || x == nil || v.doomed || v.unhold == nil {
		// Withdrawn meanwhile: the closed pause request resumes it.
		h.log.Debug("pause took effect after it was withdrawn", "seq", v.seq)
		return
	}
	if v.timer != nil {
		v.timer.Stop()
		v.timer = nil
	}
	now := h.now()
	v.state, v.since = jobPaused, now
	v.w.inFlight--
	v.pauseFor, x.preempting = nil, nil
	h.log.Debug("job paused", "seq", v.seq, "priority", v.prio, "for_seq", x.seq, "after", now.Sub(v.pauseStart))
	h.dispatchLocked()
}

func (h *Hub) pauseRefused(v *job, attempt int, err error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if v.attempt != attempt || v.unhold == nil || v.state != jobRunning {
		return
	}
	h.log.Debug("worker did not take the pause", "seq", v.seq, "err", err)
	v.unhold = nil
	v.noPause = true
	h.interruptLocked(v, "pause refused")
}

func (h *Hub) pauseTimedOut(v *job, attempt int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if v.attempt != attempt || v.pauseFor == nil || v.state != jobRunning || v.doomed {
		return
	}
	v.timer = nil
	h.log.Debug("pause timed out", "seq", v.seq, "priority", v.prio, "protected", v.protected())
	h.interruptLocked(v, "pause timed out")
}

// interruptLocked cancels v for a restart, unless it is protected: then it
// keeps a pending pause without deadline, or, when it cannot pause, the
// waiting job is unlinked to look for another victim or wait.
func (h *Hub) interruptLocked(v *job, reason string) {
	x := v.pauseFor
	if x == nil {
		return
	}
	if v.timer != nil {
		v.timer.Stop()
		v.timer = nil
	}
	if v.protected() {
		if !v.canPause() {
			v.pauseFor, x.preempting = nil, nil
			h.dispatchLocked()
		}
		return
	}
	v.doomed = true
	if v.unhold != nil {
		v.unhold()
		v.unhold = nil
	}
	h.log.Debug("job canceled for restart", "seq", v.seq, "priority", v.prio, "for_seq", x.seq, "reason", reason)
	v.cancel(errRestart)
}

// withdrawLocked ends the preemption x started, if it has not taken
// effect: x got a slot or went away.
func (h *Hub) withdrawLocked(x *job) {
	v := x.preempting
	if v == nil {
		return
	}
	x.preempting = nil
	if v.pauseFor != x {
		return
	}
	v.pauseFor = nil
	if v.doomed {
		return
	}
	if v.timer != nil {
		v.timer.Stop()
		v.timer = nil
	}
	if v.unhold != nil {
		v.unhold()
		v.unhold = nil
	}
	h.log.Debug("pause withdrawn", "seq", v.seq, "for_seq", x.seq)
}

// dropPauseLocked clears every pause state of v, which is ending.
func (h *Hub) dropPauseLocked(v *job) {
	if v.timer != nil {
		v.timer.Stop()
		v.timer = nil
	}
	if v.unhold != nil {
		v.unhold()
		v.unhold = nil
	}
	if x := v.pauseFor; x != nil {
		x.preempting = nil
		v.pauseFor = nil
	}
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

// priorityOf resolves the priority of a job of protocol for the caller in
// ctx: the caller's own (admin access: Priorities.Admin), else the
// protocol default. Unknown protocols get 0.
func (h *Hub) priorityOf(ctx context.Context, protocol string) int {
	p := h.cfg.Priorities
	id := core.IdentityFrom(ctx)
	if id.NetworkAdmin {
		if p.Admin != nil {
			return *p.Admin
		}
	} else if id.Priority != nil {
		return *id.Priority
	}
	switch protocol {
	case core.ProtocolLegacy:
		return *p.Legacy
	case core.ProtocolOpenAI:
		return *p.OpenAI
	case core.ProtocolWyoming:
		return *p.Wyoming
	}
	return 0
}
