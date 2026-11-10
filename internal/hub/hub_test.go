package hub_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/hub"
	"github.com/GRbit/whisper-hub/internal/hub/workertest"
)

// --- helpers ---

type usageLog struct {
	mu  sync.Mutex
	evs []core.UsageEvent
}

func (u *usageLog) Record(_ context.Context, ev core.UsageEvent) {
	u.mu.Lock()
	u.evs = append(u.evs, ev)
	u.mu.Unlock()
}

func (u *usageLog) last(t *testing.T) core.UsageEvent {
	t.Helper()
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.evs) == 0 {
		t.Fatal("no usage event")
	}
	return u.evs[len(u.evs)-1]
}

type authzFunc func(id *core.Identity, model string, isDefault bool) bool

func (f authzFunc) AllowModel(id *core.Identity, model string, isDefault bool) bool {
	return f(id, model, isDefault)
}

type env struct {
	hub   *hub.Hub
	usage *usageLog
}

// start builds a hub over the given fakes. Background polling is slowed to
// an hour so only the initial probe and explicit Refresh calls change
// worker state, which keeps the tests deterministic.
func start(t *testing.T, cfg hub.Config, authz core.Authorizer, limit int, workers ...*workertest.Worker) env {
	t.Helper()
	for _, w := range workers {
		cfg.Workers = append(cfg.Workers, hub.WorkerConfig{URL: w.URL, MaxInFlight: limit})
	}
	cfg.PollInterval, cfg.RetryMin, cfg.RetryMax = time.Hour, time.Hour, time.Hour
	u := &usageLog{}
	h, err := hub.New(cfg, authz, u, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	h.Refresh(context.Background())
	return env{h, u}
}

func pcm(seconds float64) core.PCM { return make(core.PCM, int(seconds*core.SampleRate)*2) }

func req(name string) *core.Request {
	return &core.Request{Audio: pcm(1), Prompt: name, Protocol: core.ProtocolOpenAI, Endpoint: "/v1/audio/transcriptions"}
}

func wantKind(t *testing.T, err error, kind core.ErrorKind) {
	t.Helper()
	if got := core.KindOf(err); err == nil || got != kind {
		t.Fatalf("want %s, got %v (%v)", kind, got, err)
	}
}

type health struct {
	code   int
	Status string `json:"status"`
	Up     int    `json:"workers_up"`
	Total  int    `json:"workers_total"`
	Queued int    `json:"queued"`
}

func healthOf(t *testing.T, h *hub.Hub) health {
	t.Helper()
	rec := httptest.NewRecorder()
	h.HealthHandler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	var hs health
	if err := json.Unmarshal(rec.Body.Bytes(), &hs); err != nil {
		t.Fatal(err)
	}
	hs.code = rec.Code
	return hs
}

// waitQueued polls the public health output until n jobs wait. Polling is
// the only way to observe "job reached the queue" from outside the hub.
func waitQueued(t *testing.T, h *hub.Hub, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for healthOf(t, h).Queued != n {
		if time.Now().After(deadline) {
			t.Fatalf("queue never reached %d (now %d)", n, healthOf(t, h).Queued)
		}
		time.Sleep(time.Millisecond)
	}
}

// gated releases one waiting job per value sent on gate.
func gated(gate <-chan struct{}) workertest.TranscribeFunc {
	return func(job *workertest.Job, s *workertest.Stream) {
		select {
		case <-gate:
		case <-job.Ctx.Done():
			return
		}
		workertest.Echo(job, s)
	}
}

func nextStarted(t *testing.T, w *workertest.Worker) *workertest.Job {
	t.Helper()
	select {
	case j := <-w.Started():
		return j
	case <-time.After(5 * time.Second):
		t.Fatal("no job reached the worker")
		return nil
	}
}

func prompt(j *workertest.Job) string {
	if j.Options.Prompt == nil {
		return ""
	}
	return *j.Options.Prompt
}

type outcome struct {
	name string
	res  *core.Result
	err  error
}

func async(h *hub.Hub, ctx context.Context, r *core.Request, out chan<- outcome) {
	go func() {
		res, err := h.Transcribe(ctx, r, nil)
		out <- outcome{r.Prompt, res, err}
	}()
}

// as returns a context whose caller has the users file priority p.
func as(p int) context.Context {
	return core.WithIdentity(context.Background(), &core.Identity{User: fmt.Sprintf("p%d", p), Models: []string{"*"}, Priority: &p})
}

func twoModelInfo() workertest.Info {
	info := workertest.DefaultInfo()
	info.Models = []workertest.ModelInfo{{ID: "tiny", Default: true}, {ID: "small"}}
	return info
}

// --- service behavior ---

func TestTranscribeStreamsSegmentsAndRecordsUsage(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	e := start(t, hub.Config{}, nil, 1, w)
	r := &core.Request{Audio: pcm(2.5), Protocol: core.ProtocolLegacy, Endpoint: "/asr", LegacyOutput: "vtt"}
	var segs []int
	ctx := core.WithIdentity(context.Background(), &core.Identity{User: "alice"})
	res, err := e.hub.Transcribe(ctx, r, func(s core.Segment) error { segs = append(segs, s.ID); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(segs, []int{0, 1, 2}) || res.Model != "tiny" || res.Engine != "faster_whisper" ||
		string(res.Rendered) != "vtt: s0 s1 s2" {
		t.Fatalf("segments %v result %+v", segs, res)
	}
	ev := e.usage.last(t)
	if ev.User != "alice" || ev.Protocol != "legacy" || ev.Endpoint != "/asr" || ev.Model != "tiny" ||
		ev.Engine != "faster_whisper" || ev.Task != core.TaskTranscribe || ev.AudioSeconds != 2.5 ||
		ev.Status != "ok" || ev.Error != "" || ev.Elapsed <= 0 || ev.Elapsed < ev.Queued || ev.Time.IsZero() {
		t.Fatalf("usage %+v", ev)
	}
}

func TestSegmentCallbackErrorCancels(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	e := start(t, hub.Config{}, nil, 1, w)
	_, err := e.hub.Transcribe(context.Background(), req("x"), func(core.Segment) error { return errors.New("gone") })
	wantKind(t, err, core.KindCanceled)
	if ev := e.usage.last(t); ev.Status != "canceled" {
		t.Fatalf("usage %+v", ev)
	}
}

func TestWorkerErrorKindsPassThrough(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(func(_ *workertest.Job, s *workertest.Stream) { s.HTTPError("internal", "Traceback ...") })
	e := start(t, hub.Config{}, nil, 1, w)
	_, err := e.hub.Transcribe(context.Background(), req("x"), nil)
	wantKind(t, err, core.KindInternal)
	if ev := e.usage.last(t); ev.Status != "internal" || ev.Error != "engine worker failed" {
		t.Fatalf("usage %+v", ev)
	}
}

func TestStartupWithLoadingWorker(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetHealthy(false)
	e := start(t, hub.Config{}, nil, 1, w)
	if e.hub.Ready() || healthOf(t, e.hub).code != http.StatusServiceUnavailable {
		t.Fatal("ready before any worker is up")
	}
	_, err := e.hub.Transcribe(context.Background(), req("x"), nil)
	wantKind(t, err, core.KindUnavailable)
	if len(e.hub.Models(context.Background())) != 0 {
		t.Fatal("models listed with no worker up")
	}
	w.SetHealthy(true)
	e.hub.Refresh(context.Background())
	if hs := healthOf(t, e.hub); !e.hub.Ready() || hs.code != http.StatusOK || hs.Up != 1 || hs.Total != 1 {
		t.Fatalf("not ready after worker came up: %+v", hs)
	}
	if _, err := e.hub.Transcribe(context.Background(), req("x"), nil); err != nil {
		t.Fatal(err)
	}
}

func TestModelResolution(t *testing.T) {
	w := workertest.New(t, twoModelInfo())
	gone := workertest.New(t, workertest.Info{Engine: "whisperx", Models: []workertest.ModelInfo{{ID: "medium", Default: true}}})
	cfg := hub.Config{Aliases: map[string]string{"big": "small"}, FailThreshold: 1}
	e := start(t, cfg, nil, 1, w, gone)
	gone.SetHealthy(false)
	e.hub.Refresh(context.Background())

	for name, want := range map[string]string{
		"": "tiny", "whisper-1": "tiny", "small": "small", "big": "small", "gpt-4o-transcribe": "tiny",
	} {
		r := req("x")
		r.Model = name
		res, err := e.hub.Transcribe(context.Background(), r, nil)
		if err != nil || res.Model != want {
			t.Errorf("model %q: got %v err %v, want %s", name, res, err, want)
		}
	}
	// Known but currently down: never a silent fallback.
	r := req("x")
	r.Model = "medium"
	_, err := e.hub.Transcribe(context.Background(), r, nil)
	wantKind(t, err, core.KindUnavailable)
}

func TestUnknownModelPolicyError(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	e := start(t, hub.Config{UnknownModel: hub.UnknownModelError}, nil, 1, w)
	r := req("x")
	r.Model = "nope"
	_, err := e.hub.Transcribe(context.Background(), r, nil)
	wantKind(t, err, core.KindModelNotFound)
	if ev := e.usage.last(t); ev.Model != "nope" || ev.Status != "model_not_found" {
		t.Fatalf("usage %+v", ev)
	}
	r.Model = "whisper-1"
	if _, err := e.hub.Transcribe(context.Background(), r, nil); err != nil {
		t.Fatalf("alias rejected under error policy: %v", err)
	}
}

func TestModelsListDefaultFirstWithAliases(t *testing.T) {
	w := workertest.New(t, twoModelInfo())
	cfg := hub.Config{DefaultModel: "small", Aliases: map[string]string{"fast": "tiny", "dangling": "large"}}
	e := start(t, cfg, nil, 1, w)
	ms := e.hub.Models(context.Background())
	if len(ms) != 2 || ms[0].ID != "small" || !ms[0].Default || ms[1].ID != "tiny" || ms[1].Default {
		t.Fatalf("models %+v", ms)
	}
	if !slices.Equal(ms[0].Aliases, []string{"whisper-1"}) || !slices.Equal(ms[1].Aliases, []string{"fast"}) {
		t.Fatalf("aliases %v %v", ms[0].Aliases, ms[1].Aliases)
	}
	if ms[0].Engine != "faster_whisper" || !ms[0].Capabilities.Translate || len(ms[0].Languages) == 0 {
		t.Fatalf("model %+v", ms[0])
	}
	res, err := e.hub.Transcribe(context.Background(), req("x"), nil)
	if err != nil || res.Model != "small" {
		t.Fatalf("configured default not used: %v %v", res, err)
	}
}

func TestAccessControl(t *testing.T) {
	w := workertest.New(t, twoModelInfo())
	var seenUser string
	onlyDefault := authzFunc(func(id *core.Identity, model string, isDefault bool) bool {
		seenUser = id.User
		return isDefault
	})
	e := start(t, hub.Config{}, onlyDefault, 1, w)
	ctx := core.WithIdentity(context.Background(), &core.Identity{User: "bob"})
	if ms := e.hub.Models(ctx); len(ms) != 1 || ms[0].ID != "tiny" || seenUser != "bob" {
		t.Fatalf("models %+v user %q", ms, seenUser)
	}
	r := req("x")
	r.Model = "small"
	_, err := e.hub.Transcribe(ctx, r, nil)
	wantKind(t, err, core.KindForbidden)
	if ev := e.usage.last(t); ev.Status != "forbidden" || ev.User != "bob" {
		t.Fatalf("usage %+v", ev)
	}
	if n := len(w.Jobs()); n != 0 {
		t.Fatalf("forbidden job reached the worker (%d jobs)", n)
	}
	// Unknown names fall back to the default, which this user may use.
	r.Model = "gpt-4o-transcribe"
	if _, err := e.hub.Transcribe(ctx, r, nil); err != nil {
		t.Fatal(err)
	}
	_, err = e.hub.DetectLanguage(ctx, &core.DetectRequest{Audio: pcm(1), Model: "small"})
	wantKind(t, err, core.KindForbidden)

	denyAll := authzFunc(func(*core.Identity, string, bool) bool { return false })
	e2 := start(t, hub.Config{}, denyAll, 1, w)
	_, err = e2.hub.Transcribe(ctx, req("x"), nil)
	wantKind(t, err, core.KindForbidden)
}

func TestCapabilityValidation(t *testing.T) {
	limited := workertest.DefaultInfo()
	limited.Capabilities = core.Capabilities{}
	limited.Languages = []string{"en"}
	w := workertest.New(t, limited)
	full := workertest.Info{Engine: "whisperx", Languages: core.Languages(),
		Models:       []workertest.ModelInfo{{ID: "dia", Default: true}},
		Capabilities: core.Capabilities{WordTimestamps: true, Diarize: true, Translate: true}}
	wd := workertest.New(t, full)
	e := start(t, hub.Config{}, nil, 1, w, wd)
	two, three, one := 2, 3, 1

	cases := []struct {
		name  string
		mod   func(r *core.Request)
		kind  *core.ErrorKind
		check func(o workertest.Options) bool
	}{
		{"legacy diarize ignored", func(r *core.Request) { r.Protocol, r.Diarize, r.MinSpeakers = core.ProtocolLegacy, true, &two },
			nil, func(o workertest.Options) bool { return !o.Diarize && o.MinSpeakers == nil }},
		{"openai diarize rejected", func(r *core.Request) { r.Diarize = true }, kind(core.KindBadRequest), nil},
		{"legacy translate downgraded", func(r *core.Request) { r.Protocol, r.Task = core.ProtocolLegacy, core.TaskTranslate },
			nil, func(o workertest.Options) bool { return o.Task == core.TaskTranscribe }},
		{"openai translate rejected", func(r *core.Request) { r.Task = core.TaskTranslate }, kind(core.KindBadRequest), nil},
		{"legacy words ignored", func(r *core.Request) { r.Protocol, r.WordTimestamps = core.ProtocolLegacy, true },
			nil, func(o workertest.Options) bool { return !o.WordTimestamps }},
		{"wyoming words rejected", func(r *core.Request) { r.Protocol, r.WordTimestamps = core.ProtocolWyoming, true }, kind(core.KindBadRequest), nil},
		{"vad dropped for everyone", func(r *core.Request) { r.VADFilter = true },
			nil, func(o workertest.Options) bool { return !o.VADFilter }},
		{"bad task", func(r *core.Request) { r.Protocol, r.Task = core.ProtocolLegacy, "summarize" }, kind(core.KindBadRequest), nil},
		{"bad output", func(r *core.Request) { r.Protocol, r.LegacyOutput = core.ProtocolLegacy, "pdf" }, kind(core.KindBadRequest), nil},
		{"empty audio", func(r *core.Request) { r.Audio = nil }, kind(core.KindBadAudio), nil},
		{"openai bogus language", func(r *core.Request) { r.Language = "xx" }, kind(core.KindBadRequest), nil},
		{"openai language not on model", func(r *core.Request) { r.Language = "de" }, kind(core.KindBadRequest), nil},
		{"legacy language passes through", func(r *core.Request) { r.Protocol, r.Language = core.ProtocolLegacy, "de" },
			nil, func(o workertest.Options) bool { return o.Language != nil && *o.Language == "de" }},
		{"openai temperature range", func(r *core.Request) { t := 2.0; r.Temperature = &t }, kind(core.KindBadRequest), nil},
		{"openai speakers range", func(r *core.Request) { r.Model, r.Diarize, r.MinSpeakers, r.MaxSpeakers = "dia", true, &three, &one },
			kind(core.KindBadRequest), nil},
		{"legacy speakers passed as is", func(r *core.Request) {
			r.Protocol, r.Model, r.Diarize, r.MinSpeakers, r.MaxSpeakers = core.ProtocolLegacy, "dia", true, &three, &one
		}, nil, func(o workertest.Options) bool { return o.Diarize && *o.MinSpeakers == 3 && *o.MaxSpeakers == 1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := len(w.Jobs()) + len(wd.Jobs())
			r := req("x")
			tc.mod(r)
			_, err := e.hub.Transcribe(context.Background(), r, nil)
			if tc.kind != nil {
				wantKind(t, err, *tc.kind)
				if len(w.Jobs())+len(wd.Jobs()) != before {
					t.Fatal("rejected job reached a worker")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var last *workertest.Job
			if r.Model == "dia" {
				last = wd.Jobs()[len(wd.Jobs())-1]
			} else {
				last = w.Jobs()[len(w.Jobs())-1]
			}
			if !tc.check(last.Options) {
				t.Fatalf("worker options %s", last.RawOptions)
			}
		})
	}
}

func kind(k core.ErrorKind) *core.ErrorKind { return &k }

func TestDetectLanguageTrimsTo30s(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetDetect(func(*workertest.Job) (string, float64, string) { return "de", 0.8, "" })
	e := start(t, hub.Config{}, nil, 1, w)
	d, err := e.hub.DetectLanguage(context.Background(), &core.DetectRequest{Audio: pcm(40), Protocol: core.ProtocolLegacy, Endpoint: "/detect-language"})
	if err != nil {
		t.Fatal(err)
	}
	if d.Language != "de" || d.Confidence != 0.8 || d.Model != "tiny" || d.Engine != "faster_whisper" {
		t.Fatalf("detection %+v", d)
	}
	if got := len(w.Jobs()[0].Audio); got != 30*core.SampleRate*2 {
		t.Fatalf("worker got %d bytes, want 30 s", got)
	}
	if ev := e.usage.last(t); ev.AudioSeconds != 30 || ev.Status != "ok" || ev.Endpoint != "/detect-language" {
		t.Fatalf("usage %+v", ev)
	}
	_, err = e.hub.DetectLanguage(context.Background(), &core.DetectRequest{})
	wantKind(t, err, core.KindBadAudio)
}

// --- scheduling, observed through the Service and the fake worker ---

func TestConcurrencyLimit(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{}, nil, 2, w)
	out := make(chan outcome, 3)
	for _, n := range []string{"a", "b", "c"} {
		async(e.hub, context.Background(), req(n), out)
	}
	nextStarted(t, w)
	nextStarted(t, w)
	waitQueued(t, e.hub, 1)
	for range 3 {
		gate <- struct{}{}
	}
	for range 3 {
		if o := <-out; o.err != nil {
			t.Fatal(o.err)
		}
	}
	if got := w.MaxInFlight(); got != 2 {
		t.Fatalf("max in flight %d, want 2", got)
	}
}

func TestPriorityOrderFIFOWithinPriority(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{PreemptTimeout: -1}, nil, 1, w)
	out := make(chan outcome, 8)
	async(e.hub, context.Background(), req("holder"), out)
	nextStarted(t, w)
	queue := []struct {
		name string
		p    int
	}{
		{"batch1", -3}, {"normal1", 50}, {"batch2", -3},
		{"inter1", 100}, {"normal2", 50}, {"inter2", 100},
	}
	for i, q := range queue {
		async(e.hub, as(q.p), req(q.name), out)
		waitQueued(t, e.hub, i+1)
	}
	for _, want := range []string{"inter1", "inter2", "normal1", "normal2", "batch1", "batch2"} {
		gate <- struct{}{}
		if got := prompt(nextStarted(t, w)); got != want {
			t.Fatalf("started %s, want %s", got, want)
		}
	}
	gate <- struct{}{}
	for range 7 {
		<-out
	}
}

// Waiting does not raise priority: a batch job that has waited longer
// still starts after an interactive one that arrived later.
func TestWaitingDoesNotRaisePriority(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{PreemptTimeout: -1}, nil, 1, w)
	out := make(chan outcome, 3)
	async(e.hub, context.Background(), req("holder"), out)
	nextStarted(t, w)
	async(e.hub, as(0), req("batch"), out)
	waitQueued(t, e.hub, 1)
	time.Sleep(60 * time.Millisecond)
	async(e.hub, as(100), req("interactive"), out)
	waitQueued(t, e.hub, 2)
	gate <- struct{}{}
	if got := prompt(nextStarted(t, w)); got != "interactive" {
		t.Fatalf("first %s, want interactive", got)
	}
	gate <- struct{}{}
	gate <- struct{}{}
	for range 3 {
		<-out
	}
}

func TestCancelWhileQueued(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{PreemptTimeout: -1}, nil, 1, w)
	out := make(chan outcome, 3)
	async(e.hub, context.Background(), req("holder"), out)
	nextStarted(t, w)
	ctx, cancel := context.WithCancel(as(100))
	async(e.hub, ctx, req("canceled"), out)
	waitQueued(t, e.hub, 1)
	async(e.hub, context.Background(), req("next"), out)
	waitQueued(t, e.hub, 2)
	cancel()
	o := <-out
	if o.name != "canceled" {
		t.Fatalf("got %s first", o.name)
	}
	wantKind(t, o.err, core.KindCanceled)
	if hs := healthOf(t, e.hub); hs.Queued != 1 {
		t.Fatalf("canceled job still queued: %+v", hs)
	}
	if ev := e.usage.last(t); ev.Status != "canceled" || ev.Queued <= 0 {
		t.Fatalf("usage %+v", ev)
	}
	gate <- struct{}{}
	if got := prompt(nextStarted(t, w)); got != "next" {
		t.Fatalf("slot went to %s", got)
	}
	gate <- struct{}{}
	<-out
	<-out
	for _, j := range w.Jobs() {
		if prompt(j) == "canceled" {
			t.Fatal("canceled job reached the worker")
		}
	}
}

func TestQueueOverflow(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{QueueLength: 2, PreemptTimeout: -1}, nil, 1, w)
	out := make(chan outcome, 8)
	async(e.hub, context.Background(), req("holder"), out)
	nextStarted(t, w)
	for i, n := range []string{"b1", "b2"} {
		async(e.hub, as(0), req(n), out)
		waitQueued(t, e.hub, i+1)
	}
	_, err := e.hub.Transcribe(as(0), req("b3"), nil)
	wantKind(t, err, core.KindUnavailable)

	// Higher priority evicts the newest lower priority job instead.
	async(e.hub, as(100), req("i1"), out)
	o := <-out
	if o.name != "b2" {
		t.Fatalf("evicted %s, want b2", o.name)
	}
	wantKind(t, o.err, core.KindUnavailable)
	if hs := healthOf(t, e.hub); hs.Queued != 2 {
		t.Fatalf("queued %d", hs.Queued)
	}
	for range 3 {
		gate <- struct{}{}
	}
	for range 3 {
		<-out
	}
}

func TestSpreadAcrossWorkersOfOneModel(t *testing.T) {
	w1, w2 := workertest.New(t, workertest.DefaultInfo()), workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w1.SetTranscribe(gated(gate))
	w2.SetTranscribe(gated(gate))
	e := start(t, hub.Config{}, nil, 1, w1, w2)
	out := make(chan outcome, 2)
	async(e.hub, context.Background(), req("a"), out)
	async(e.hub, context.Background(), req("b"), out)
	nextStarted(t, w1)
	nextStarted(t, w2)
	close(gate)
	<-out
	<-out
}

func TestNoHeadOfLineBlockingAcrossModels(t *testing.T) {
	wa := workertest.New(t, workertest.Info{Engine: "e", Models: []workertest.ModelInfo{{ID: "a", Default: true}}})
	wb := workertest.New(t, workertest.Info{Engine: "e", Models: []workertest.ModelInfo{{ID: "b", Default: true}}})
	gate := make(chan struct{})
	wa.SetTranscribe(gated(gate))
	e := start(t, hub.Config{PreemptTimeout: -1}, nil, 1, wa, wb)
	out := make(chan outcome, 3)
	a1 := req("a1")
	a1.Model = "a"
	async(e.hub, context.Background(), a1, out)
	nextStarted(t, wa)
	a2 := req("a2")
	a2.Model = "a"
	async(e.hub, as(100), a2, out)
	waitQueued(t, e.hub, 1)
	b := req("b")
	b.Model = "b"
	if _, err := e.hub.Transcribe(as(0), b, nil); err != nil {
		t.Fatalf("job for an idle worker was blocked: %v", err)
	}
	close(gate)
	<-out
	<-out
}

func TestQueuedJobFailsWhenWorkerGoesDown(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{FailThreshold: 1}, nil, 1, w)
	out := make(chan outcome, 2)
	async(e.hub, context.Background(), req("holder"), out)
	nextStarted(t, w)
	async(e.hub, context.Background(), req("queued"), out)
	waitQueued(t, e.hub, 1)
	w.SetHealthy(false)
	e.hub.Refresh(context.Background())
	o := <-out
	if o.name != "queued" {
		t.Fatalf("got %s", o.name)
	}
	wantKind(t, o.err, core.KindUnavailable)
	if e.hub.Ready() {
		t.Fatal("ready with the only worker down")
	}
	// The running job is not interrupted by the health flap.
	gate <- struct{}{}
	if o := <-out; o.err != nil {
		t.Fatalf("in-flight job failed: %v", o.err)
	}
}

func TestConfigValidation(t *testing.T) {
	for _, cfg := range []hub.Config{
		{},
		{Workers: []hub.WorkerConfig{{URL: "not a url"}}},
		{Workers: []hub.WorkerConfig{{URL: "http://a:1"}, {URL: "http://a:1"}}},
		{Workers: []hub.WorkerConfig{{URL: "http://a:1"}}, UnknownModel: "maybe"},
	} {
		if h, err := hub.New(cfg, nil, nil, nil); err == nil {
			h.Close()
			t.Errorf("config %+v accepted", cfg)
		}
	}
}

// Capabilities of a model served by several workers are ANDed, so a job
// valid for the listed model is valid on whichever worker it lands.
func TestModelsMergeCapabilitiesAndDefaultLoss(t *testing.T) {
	rich := workertest.Info{Engine: "whisperx", Models: []workertest.ModelInfo{{ID: "m", Default: true}},
		Capabilities: core.Capabilities{Diarize: true, Translate: true}}
	poor := rich
	poor.Capabilities = core.Capabilities{Translate: true}
	other := workertest.Info{Engine: "faster_whisper", Models: []workertest.ModelInfo{{ID: "tiny", Default: true}}}
	w1, w2, w3 := workertest.New(t, rich), workertest.New(t, poor), workertest.New(t, other)
	e := start(t, hub.Config{DefaultModel: "m", FailThreshold: 1}, nil, 1, w1, w2, w3)
	ms := e.hub.Models(context.Background())
	if len(ms) != 2 || ms[0].ID != "m" || ms[0].Capabilities.Diarize || !ms[0].Capabilities.Translate {
		t.Fatalf("models %+v", ms)
	}
	r := req("x")
	r.Diarize = true
	_, err := e.hub.Transcribe(context.Background(), r, nil)
	wantKind(t, err, core.KindBadRequest)

	w1.SetHealthy(false)
	w2.SetHealthy(false)
	e.hub.Refresh(context.Background())
	if ms := e.hub.Models(context.Background()); len(ms) != 1 || ms[0].ID != "tiny" || ms[0].Default {
		t.Fatalf("after default model lost: %+v", ms)
	}
	if e.hub.Ready() {
		t.Fatal("ready although the configured default is gone")
	}
	_, err = e.hub.Transcribe(context.Background(), req("x"), nil)
	wantKind(t, err, core.KindUnavailable)
}

// Canceling the caller of a running job aborts the worker request (the
// fake sees its request context end) and frees the slot.
func TestCancelRunningJobReachesWorkerAndFreesSlot(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	gate := make(chan struct{})
	w.SetTranscribe(gated(gate))
	e := start(t, hub.Config{}, nil, 1, w)
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan outcome, 1)
	async(e.hub, ctx, req("running"), out)
	job := nextStarted(t, w)
	cancel()
	wantKind(t, (<-out).err, core.KindCanceled)
	select {
	case <-job.Ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("worker request not canceled")
	}
	w.SetTranscribe(nil)
	if _, err := e.hub.Transcribe(context.Background(), req("after"), nil); err != nil {
		t.Fatalf("slot not freed: %v", err)
	}
}
