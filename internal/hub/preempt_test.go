package hub_test

// Preemption tests. The fake worker runs each job in a script; how a test
// knows what the scheduler did:
//
//   - A pause took effect: with max_in_flight 1 the hub sends the higher job
//     to the worker only after the worker's stream reported the lower job
//     paused, so the higher job arriving at the fake (nextStarted) while
//     the lower one is still open proves it. Scripts built on Stream.Await
//     write that paused line whenever a pause is held, like the real worker
//     at a pause point.
//   - The timeout path: the lower job's script ignores pauses (gated), so a
//     cancel is the only way the higher job can ever run; PreemptTimeout is
//     a nanosecond so it fires at once. Where the cancel must NOT happen, the
//     test waits for the hub's own "asking job to pause" debug line, written
//     when the hub decides how the job may be interrupted, so the check runs
//     after that decision, not before it.
//   - Long timeouts (an hour) make the pause the only possible outcome.
//
// No sleep decides an outcome; the few sleeps establish a minimum duration
// for the accounting checks.

import (
	"bytes"
	"context"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/hub"
	"github.com/GRbit/whisper-hub/internal/hub/workertest"
)

type logBuf struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *logBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *logBuf) String() string { b.mu.Lock(); defer b.mu.Unlock(); return b.buf.String() }

// waitLog waits until the hub logged msg n times in total.
func (b *logBuf) waitLog(t *testing.T, msg string, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for strings.Count(b.String(), "msg=\""+msg+"\"") < n {
		if time.Now().After(deadline) {
			t.Fatalf("hub never logged %q %d times; log:\n%s", msg, n, b.String())
		}
		time.Sleep(time.Millisecond)
	}
}

// startLogged is start with the hub's debug log captured.
func startLogged(t *testing.T, cfg hub.Config, limit int, workers ...*workertest.Worker) (env, *logBuf) {
	t.Helper()
	for _, w := range workers {
		cfg.Workers = append(cfg.Workers, hub.WorkerConfig{URL: w.URL, MaxInFlight: limit})
	}
	cfg.PollInterval, cfg.RetryMin, cfg.RetryMax = time.Hour, time.Hour, time.Hour
	u := &usageLog{}
	lb := &logBuf{}
	h, err := hub.New(cfg, nil, u, slog.New(slog.NewTextHandler(lb, &slog.HandlerOptions{Level: slog.LevelDebug})))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	h.Refresh(context.Background())
	return env{h, u}, lb
}

// byPrompt runs a script chosen by the job's prompt; jobs without a script
// run Echo at once.
func byPrompt(scripts map[string]workertest.TranscribeFunc) workertest.TranscribeFunc {
	return func(job *workertest.Job, s *workertest.Stream) {
		if f, ok := scripts[prompt(job)]; ok {
			f(job, s)
			return
		}
		workertest.Echo(job, s)
	}
}

func started(w *workertest.Worker) []string {
	var out []string
	for _, j := range w.Jobs() {
		out = append(out, prompt(j))
	}
	return out
}

func (u *usageLog) byUser() map[string]core.UsageEvent {
	u.mu.Lock()
	defer u.mu.Unlock()
	m := map[string]core.UsageEvent{}
	for _, ev := range u.evs {
		m[ev.User] = ev
	}
	return m
}

func (u *usageLog) count() int { u.mu.Lock(); defer u.mu.Unlock(); return len(u.evs) }

func echoText(seconds float64) string {
	var b strings.Builder
	for i := 0; float64(i) < seconds; i++ {
		b.WriteString(" s" + string(rune('0'+i)))
	}
	return b.String()
}

// A higher job pauses the running lower one, runs, and the lower one then
// continues where it stopped with an unchanged result. Its time paused is
// queue time in the usage record.
func TestPauseLetsHigherJobRunFirst(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low, high := make(chan struct{}), make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{
		"low": workertest.Pausing(low), "high": workertest.Pausing(high),
	}))
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 1, w)
	out := make(chan outcome, 2)
	lr := req("low")
	lr.Audio = pcm(3)
	async(e.hub, as(10), lr, out)
	lj := nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	if got := prompt(nextStarted(t, w)); got != "high" {
		t.Fatalf("started %s, want high", got)
	}
	if n := w.Paused(lj.ID); n != 1 {
		t.Fatalf("low paused %d times", n)
	}
	time.Sleep(50 * time.Millisecond) // a minimum pause length for the accounting check
	close(high)
	if o := <-out; o.name != "high" || o.err != nil {
		t.Fatalf("first result %s %v", o.name, o.err)
	}
	close(low)
	o := <-out
	if o.err != nil || o.res.Text != echoText(3) || len(o.res.Segments) != 3 {
		t.Fatalf("low result %+v %v", o.res, o.err)
	}
	if got := started(w); !slices.Equal(got, []string{"low", "high"}) {
		t.Fatalf("worker jobs %v: low must not restart", got)
	}
	ev := e.usage.byUser()["p10"]
	if ev.Status != "ok" || ev.Queued < 50*time.Millisecond || ev.Elapsed < ev.Queued {
		t.Fatalf("low usage %+v", ev)
	}
	if e.usage.count() != 2 {
		t.Fatalf("%d usage records, want 2", e.usage.count())
	}
}

// A job that ignores the pause is canceled when PreemptTimeout runs out
// and restarts from the beginning later; its caller sees one normal
// result and one usage record.
func TestPauseTimeoutRestartsJob(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": gated(low)}))
	e, lb := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond}, 1, w)
	out := make(chan outcome, 2)
	async(e.hub, as(10), req("low"), out)
	first := nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	if got := prompt(nextStarted(t, w)); got != "high" {
		t.Fatalf("started %s, want high", got)
	}
	select {
	case <-first.Ctx.Done():
	default:
		t.Fatal("first low attempt still running at the worker")
	}
	if o := <-out; o.name != "high" || o.err != nil {
		t.Fatalf("first result %s %v", o.name, o.err)
	}
	second := nextStarted(t, w)
	if prompt(second) != "low" || second.ID == first.ID {
		t.Fatalf("restart %q id %q (first %q)", prompt(second), second.ID, first.ID)
	}
	close(low)
	if o := <-out; o.err != nil || o.res.Text != echoText(1) {
		t.Fatalf("low result %+v %v", o.res, o.err)
	}
	if ev := e.usage.byUser()["p10"]; ev.Status != "ok" || e.usage.count() != 2 {
		t.Fatalf("usage %+v (%d records)", ev, e.usage.count())
	}
	lb.waitLog(t, "job canceled for restart", 1)
	lb.waitLog(t, "job restarting", 1)
}

// A worker that refuses to pause is not waited for.
func TestPauseRefusedRestartsAtOnce(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": workertest.Pausing(low)}))
	w.SetPauseRefused(true)
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 1, w)
	out := make(chan outcome, 2)
	async(e.hub, as(10), req("low"), out)
	nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	if got := prompt(nextStarted(t, w)); got != "high" {
		t.Fatalf("started %s, want high", got)
	}
	<-out
	close(low)
	if o := <-out; o.err != nil {
		t.Fatal(o.err)
	}
	if got := started(w); !slices.Equal(got, []string{"low", "high", "low"}) {
		t.Fatalf("worker jobs %v", got)
	}
}

// A job that already handed segments to its caller is never canceled for
// a restart: the caller would see output twice. When it does not pause,
// the higher job waits for it.
func TestDeliveredJobIsNotRestarted(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": func(job *workertest.Job, s *workertest.Stream) {
		s.Info(job.Model, job.Engine, "en", 1)
		s.Segment(core.Segment{ID: 0, Text: " a"})
		select {
		case <-low:
		case <-job.Ctx.Done():
			return
		}
		s.Done(" a", "en", nil)
	}}))
	e, lb := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond}, 1, w)
	out := make(chan outcome, 2)
	got := make(chan core.Segment, 4)
	go func() {
		res, err := e.hub.Transcribe(as(10), req("low"), func(s core.Segment) error { got <- s; return nil })
		out <- outcome{"low", res, err}
	}()
	nextStarted(t, w)
	<-got
	async(e.hub, as(90), req("high"), out)
	// Logged once the hub decided how to interrupt it (protected=true:
	// pause only, no deadline).
	lb.waitLog(t, "asking job to pause", 1)
	close(low)
	if o := <-out; o.name != "low" || o.err != nil {
		t.Fatalf("first result %s %v", o.name, o.err)
	}
	if o := <-out; o.name != "high" || o.err != nil {
		t.Fatalf("second result %s %v", o.name, o.err)
	}
	if got := started(w); !slices.Equal(got, []string{"low", "high"}) {
		t.Fatalf("worker jobs %v", got)
	}
}

// A streaming job may still be paused; its caller just sees a gap.
func TestDeliveredJobCanPause(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low, high := make(chan struct{}), make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{
		"low": func(job *workertest.Job, s *workertest.Stream) {
			s.Info(job.Model, job.Engine, "en", 2)
			s.Segment(core.Segment{ID: 0, Text: " a"})
			if !s.Await(job, low) {
				return
			}
			s.Segment(core.Segment{ID: 1, Text: " b"})
			s.Done(" a b", "en", nil)
		},
		"high": workertest.Pausing(high),
	}))
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond}, 1, w)
	out := make(chan outcome, 2)
	var segs []string
	first := make(chan struct{})
	go func() {
		res, err := e.hub.Transcribe(as(10), req("low"), func(s core.Segment) error {
			segs = append(segs, s.Text)
			if s.ID == 0 {
				close(first)
			}
			return nil
		})
		out <- outcome{"low", res, err}
	}()
	nextStarted(t, w)
	<-first
	async(e.hub, as(90), req("high"), out)
	if got := prompt(nextStarted(t, w)); got != "high" {
		t.Fatalf("started %s", got)
	}
	close(high)
	<-out
	close(low)
	o := <-out
	if o.err != nil || o.res.Text != " a b" || !slices.Equal(segs, []string{" a", " b"}) {
		t.Fatalf("low %+v %v segments %v", o.res, o.err, segs)
	}
}

// Restarting is bounded: a job is canceled for a restart at most once.
// Afterwards it is treated like a job that delivered output: it may pause,
// but a higher job waits for it rather than cancel it again.
func TestRestartHappensAtMostOnce(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": gated(low)}))
	e, lb := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond}, 1, w)
	out := make(chan outcome, 3)
	async(e.hub, as(10), req("low"), out)
	nextStarted(t, w)
	async(e.hub, as(90), req("high1"), out)
	nextStarted(t, w) // high1, after low was canceled
	<-out
	nextStarted(t, w) // low again
	async(e.hub, as(90), req("high2"), out)
	lb.waitLog(t, "asking job to pause", 2)
	close(low)
	if o := <-out; o.name != "low" || o.err != nil {
		t.Fatalf("got %s %v, want low first", o.name, o.err)
	}
	<-out
	if got := started(w); !slices.Equal(got, []string{"low", "high1", "low", "high2"}) {
		t.Fatalf("worker jobs %v", got)
	}
}

func TestCancelWhilePaused(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	high := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{
		"low": workertest.Pausing(make(chan struct{})), "high": workertest.Pausing(high),
	}))
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 1, w)
	out := make(chan outcome, 3)
	ctx, cancel := context.WithCancel(as(10))
	async(e.hub, ctx, req("low"), out)
	lj := nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	nextStarted(t, w)
	cancel()
	o := <-out
	if o.name != "low" {
		t.Fatalf("got %s", o.name)
	}
	wantKind(t, o.err, core.KindCanceled)
	select {
	case <-lj.Ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("paused job not canceled at the worker")
	}
	if ev := e.usage.byUser()["p10"]; ev.Status != "canceled" {
		t.Fatalf("usage %+v", ev)
	}
	close(high)
	<-out
	// The slot accounting survived: the next job runs at once.
	if _, err := e.hub.Transcribe(as(10), req("next"), nil); err != nil {
		t.Fatal(err)
	}
}

func TestCancelWhileWaitingForRestart(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	high := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{
		"low": gated(make(chan struct{})), "high": gated(high),
	}))
	e, lb := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond}, 1, w)
	out := make(chan outcome, 2)
	ctx, cancel := context.WithCancel(as(10))
	async(e.hub, ctx, req("low"), out)
	nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	nextStarted(t, w)
	lb.waitLog(t, "job restarting", 1)
	waitQueued(t, e.hub, 1)
	cancel()
	o := <-out
	if o.name != "low" {
		t.Fatalf("got %s", o.name)
	}
	wantKind(t, o.err, core.KindCanceled)
	waitQueued(t, e.hub, 0)
	close(high)
	<-out
	if got := started(w); !slices.Equal(got, []string{"low", "high"}) {
		t.Fatalf("worker jobs %v", got)
	}
}

// Jobs that already ran (paused, or waiting to restart) are not counted
// against queue_length and are never evicted: only new arrivals are.
func TestQueueFullNeverEvictsJobsThatRan(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low, high := make(chan struct{}), make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": gated(low), "high": gated(high)}))
	e, lb := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond, QueueLength: 1}, 1, w)
	out := make(chan outcome, 8)
	async(e.hub, as(10), req("low"), out)
	nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	nextStarted(t, w)
	lb.waitLog(t, "job restarting", 1)
	waitQueued(t, e.hub, 1) // low, waiting to restart
	async(e.hub, as(50), req("n1"), out)
	waitQueued(t, e.hub, 2)
	async(e.hub, as(60), req("n2"), out)
	o := <-out
	if o.name != "n1" {
		t.Fatalf("evicted %s, want n1", o.name)
	}
	wantKind(t, o.err, core.KindUnavailable)
	waitQueued(t, e.hub, 2)
	close(high)
	close(low)
	for range 3 {
		if o := <-out; o.err != nil {
			t.Fatalf("%s: %v", o.name, o.err)
		}
	}
}

// A worker that fails while a job is paused on it fails that job like a
// running one: the job's state lives in that worker process.
func TestWorkerFailureDuringPause(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{
		"low": workertest.Pausing(make(chan struct{})), "high": workertest.Pausing(make(chan struct{})),
	}))
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 1, w)
	out := make(chan outcome, 2)
	async(e.hub, as(10), req("low"), out)
	nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	nextStarted(t, w)
	w.CloseClientConnections()
	for range 2 {
		o := <-out
		if core.KindOf(o.err) != core.KindUnavailable && core.KindOf(o.err) != core.KindInternal {
			t.Fatalf("%s: %v", o.name, o.err)
		}
	}
	w.SetTranscribe(nil)
	e.hub.Refresh(context.Background())
	if _, err := e.hub.Transcribe(as(10), req("next"), nil); err != nil {
		t.Fatalf("slot accounting broken after the failure: %v", err)
	}
}

// A paused job competes for its worker like a waiting one: by priority,
// then arrival. Here a job that arrived after it with the same priority
// waits behind it, and one with a higher priority goes first.
func TestPausedJobResumesInPolicyOrder(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gates := map[string]chan struct{}{"low": make(chan struct{}), "high": make(chan struct{}),
		"mid": make(chan struct{}), "late": make(chan struct{})}
	scripts := map[string]workertest.TranscribeFunc{}
	for k, g := range gates {
		scripts[k] = workertest.Pausing(g)
	}
	w.SetTranscribe(byPrompt(scripts))
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 1, w)
	out := make(chan outcome, 4)
	async(e.hub, as(10), req("low"), out)
	nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	nextStarted(t, w)
	async(e.hub, as(10), req("late"), out)
	waitQueued(t, e.hub, 1)
	async(e.hub, as(50), req("mid"), out)
	waitQueued(t, e.hub, 2)
	for _, n := range []string{"high", "mid", "low", "late"} {
		close(gates[n])
		if o := <-out; o.name != n || o.err != nil {
			t.Fatalf("got %s %v, want %s", o.name, o.err, n)
		}
	}
	if got := started(w); !slices.Equal(got, []string{"low", "high", "mid", "late"}) {
		t.Fatalf("worker jobs %v", got)
	}
}

// Of several running lower jobs, the lowest is asked to pause.
func TestLowestRunningJobIsPaused(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gates := map[string]chan struct{}{"a": make(chan struct{}), "b": make(chan struct{}), "high": make(chan struct{})}
	scripts := map[string]workertest.TranscribeFunc{}
	for k, g := range gates {
		scripts[k] = workertest.Pausing(g)
	}
	w.SetTranscribe(byPrompt(scripts))
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 2, w)
	out := make(chan outcome, 3)
	async(e.hub, as(20), req("a"), out)
	aj := nextStarted(t, w)
	async(e.hub, as(10), req("b"), out)
	bj := nextStarted(t, w)
	async(e.hub, as(90), req("high"), out)
	nextStarted(t, w)
	if w.Paused(aj.ID) != 0 || w.Paused(bj.ID) != 1 {
		t.Fatalf("paused a=%d b=%d, want only b", w.Paused(aj.ID), w.Paused(bj.ID))
	}
	for _, g := range gates {
		close(g)
	}
	for range 3 {
		if o := <-out; o.err != nil {
			t.Fatal(o.err)
		}
	}
}

// Equal priority never interrupts, and a negative PreemptTimeout turns
// interruption off.
func TestNoPreemption(t *testing.T) {
	for _, tc := range []struct {
		name          string
		timeout       time.Duration
		lowP, highP   int
		expectRequest bool
	}{
		{"equal priority", time.Nanosecond, 50, 50, false},
		{"preemption off", -1, 10, 90, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := workertest.New(t, workertest.DefaultInfo())
			low := make(chan struct{})
			w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": workertest.Pausing(low)}))
			e, _ := startLogged(t, hub.Config{PreemptTimeout: tc.timeout}, 1, w)
			out := make(chan outcome, 2)
			async(e.hub, as(tc.lowP), req("low"), out)
			nextStarted(t, w)
			async(e.hub, as(tc.highP), req("high"), out)
			waitQueued(t, e.hub, 1)
			close(low)
			if o := <-out; o.name != "low" {
				t.Fatalf("got %s first", o.name)
			}
			<-out
			if n := len(w.PauseRequests()); n != 0 {
				t.Fatalf("%d pause requests", n)
			}
		})
	}
}

// When the waiting job goes away before the pause took effect, the pause
// request is withdrawn and the running job carries on undisturbed.
func TestPauseWithdrawnWhenWaiterLeaves(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	low := make(chan struct{})
	w.SetTranscribe(byPrompt(map[string]workertest.TranscribeFunc{"low": gated(low)}))
	e, lb := startLogged(t, hub.Config{PreemptTimeout: time.Hour}, 1, w)
	out := make(chan outcome, 2)
	async(e.hub, as(10), req("low"), out)
	nextStarted(t, w)
	ctx, cancel := context.WithCancel(as(90))
	async(e.hub, ctx, req("high"), out)
	waitQueued(t, e.hub, 1)
	lb.waitLog(t, "asking job to pause", 1)
	cancel()
	wantKind(t, (<-out).err, core.KindCanceled)
	lb.waitLog(t, "pause withdrawn", 1)
	close(low)
	if o := <-out; o.err != nil {
		t.Fatal(o.err)
	}
	if got := started(w); !slices.Equal(got, []string{"low"}) {
		t.Fatalf("worker jobs %v", got)
	}
}

// Language detection has no pause point: it is canceled and restarted.
func TestDetectIsRestartedNotPaused(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	var mu sync.Mutex
	calls := 0
	w.SetDetect(func(job *workertest.Job) (string, float64, string) {
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()
		if first {
			<-job.Ctx.Done()
			return "", 0, "internal"
		}
		return "de", 0.9, ""
	})
	e, _ := startLogged(t, hub.Config{PreemptTimeout: time.Nanosecond}, 1, w)
	det := make(chan error, 1)
	go func() {
		d, err := e.hub.DetectLanguage(as(10), &core.DetectRequest{Audio: pcm(1), Protocol: core.ProtocolLegacy})
		if err == nil && d.Language != "de" {
			err = context.DeadlineExceeded
		}
		det <- err
	}()
	nextStarted(t, w)
	if _, err := e.hub.Transcribe(as(90), req("high"), nil); err != nil {
		t.Fatal(err)
	}
	if err := <-det; err != nil {
		t.Fatal(err)
	}
	if n := len(w.PauseRequests()); n != 0 {
		t.Fatalf("%d pause requests for a detection", n)
	}
}

// Priorities: the caller's own wins, admin access has its own optional
// setting, everything else gets the protocol default.
func TestPriorityResolution(t *testing.T) {
	ten, twenty, thirty := 10, 20, 30
	for _, tc := range []struct {
		name  string
		admin *int
		want  []string
	}{
		{"no admin setting", nil, []string{"wyoming", "user25", "openai", "admin-legacy"}},
		{"admin setting", &thirty, []string{"admin-legacy", "wyoming", "user25", "openai"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := workertest.New(t, workertest.DefaultInfo())
			gate := make(chan struct{})
			w.SetTranscribe(gated(gate))
			cfg := hub.Config{PreemptTimeout: -1, Priorities: hub.Priorities{Legacy: &ten, OpenAI: &twenty, Wyoming: &thirty,
				Admin: tc.admin}}
			e, _ := startLogged(t, cfg, 1, w)
			out := make(chan outcome, 5)
			async(e.hub, as(1000), req("holder"), out)
			nextStarted(t, w)
			adminLegacy := req("admin-legacy")
			adminLegacy.Protocol = core.ProtocolLegacy
			wy := req("wyoming")
			wy.Protocol = core.ProtocolWyoming
			plain := core.WithIdentity(context.Background(), &core.Identity{User: "bob", Models: []string{"*"}})
			wyCtx := core.WithIdentity(context.Background(), &core.Identity{User: core.WyomingUser, Models: []string{"*"}})
			u25 := req("user25")
			u25.Protocol = core.ProtocolLegacy
			for i, j := range []struct {
				ctx context.Context
				r   *core.Request
			}{
				{context.Background(), adminLegacy}, {plain, req("openai")}, {as(25), u25}, {wyCtx, wy},
			} {
				async(e.hub, j.ctx, j.r, out)
				waitQueued(t, e.hub, i+1)
			}
			for _, want := range tc.want {
				gate <- struct{}{}
				if got := prompt(nextStarted(t, w)); got != want {
					t.Fatalf("started %s, want %s", got, want)
				}
			}
			gate <- struct{}{}
			for range 5 {
				<-out
			}
		})
	}
}

func TestDefaultPriorities(t *testing.T) {
	p := hub.Config{}.Resolved().Priorities
	if *p.Legacy != 50 || *p.OpenAI != 50 || *p.Wyoming != 100 || p.Admin != nil {
		t.Fatalf("defaults %+v", p)
	}
	if d := (hub.Config{}).Resolved().PreemptTimeout; d != 5*time.Second {
		t.Fatalf("preempt_timeout default %v", d)
	}
}
