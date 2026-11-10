// Package hub implements core.Service on top of one or more engine workers:
// worker discovery and health, model resolution and access control,
// priority scheduling and usage accounting.
package hub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/hub/internal/workerclient"
)

// Config configures the hub. Zero values select the documented defaults.
type Config struct {
	// Workers lists the engine workers. At least one is required.
	Workers []WorkerConfig `yaml:"workers"`
	// DefaultModel is the model id used for empty or unknown model names.
	// Default: the default model of the first healthy worker in Workers
	// order.
	DefaultModel string `yaml:"default_model"`
	// Aliases maps public names to model ids; an empty target means the
	// default model. "whisper-1" -> default is added unless overridden.
	Aliases map[string]string `yaml:"aliases"`
	// UnknownModel is the policy for model names that are neither a served
	// id, an alias, nor a model seen earlier: "default" (fall back to the
	// default model, the default) or "error" (KindModelNotFound).
	UnknownModel string `yaml:"unknown_model"`
	// QueueLength caps the jobs waiting for a worker, across all workers.
	// Default 100.
	QueueLength int `yaml:"queue_length"`
	// PriorityAging raises the priority of a waiting job by one level per
	// interval waited, so batch jobs cannot starve forever under constant
	// interactive load. Default 5m; negative disables aging (strict
	// priority, batch may starve).
	PriorityAging time.Duration `yaml:"priority_aging"`
	// PollInterval is how often a healthy worker is re-probed. Default 15s.
	PollInterval time.Duration `yaml:"poll_interval"`
	// RetryMin and RetryMax bound the exponential backoff for probing an
	// unhealthy or still loading worker. Defaults 1s and 30s.
	RetryMin time.Duration `yaml:"retry_min"`
	RetryMax time.Duration `yaml:"retry_max"`
	// FailThreshold is how many consecutive failed probes take a healthy
	// worker out of rotation; a busy Python worker can miss one. Default 2.
	FailThreshold int `yaml:"fail_threshold"`
	// ConnectTimeout bounds TCP connection setup to a worker. Default 5s.
	ConnectTimeout time.Duration `yaml:"connect_timeout"`
	// FirstByteTimeout bounds the wait for the first response line of a
	// job; it must cover lazy model loading. There is deliberately no
	// overall job deadline: a long file can take an hour. Default 10m.
	FirstByteTimeout time.Duration `yaml:"first_byte_timeout"`
	// ProbeTimeout bounds one /healthz or /v1/info call. Default 30s.
	ProbeTimeout time.Duration `yaml:"probe_timeout"`
}

// WorkerConfig describes one engine worker.
type WorkerConfig struct {
	// URL is the worker base URL, e.g. "http://worker:9100".
	URL string `yaml:"url"`
	// MaxInFlight is how many jobs the hub sends to this worker at once.
	// Default 1: the worker runs one job per model anyway.
	MaxInFlight int `yaml:"max_in_flight"`
}

// Unknown-model policies for Config.UnknownModel.
const (
	UnknownModelDefault = "default"
	UnknownModelError   = "error"
)

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it. It does not validate; New does.
func (c Config) Resolved() Config {
	if c.UnknownModel == "" {
		c.UnknownModel = UnknownModelDefault
	}
	aliases := map[string]string{"whisper-1": ""}
	for k, v := range c.Aliases {
		aliases[k] = v
	}
	c.Aliases = aliases
	if c.QueueLength <= 0 {
		c.QueueLength = 100
	}
	if c.PriorityAging == 0 {
		c.PriorityAging = 5 * time.Minute
	}
	if c.PollInterval <= 0 {
		c.PollInterval = 15 * time.Second
	}
	if c.RetryMin <= 0 {
		c.RetryMin = time.Second
	}
	if c.RetryMax <= 0 {
		c.RetryMax = 30 * time.Second
	}
	c.RetryMax = max(c.RetryMax, c.RetryMin)
	if c.FailThreshold <= 0 {
		c.FailThreshold = 2
	}
	// The worker client owns these defaults; the hub only renames them.
	wc := workerclient.Config{ConnectTimeout: c.ConnectTimeout, FirstByteTimeout: c.FirstByteTimeout,
		RequestTimeout: c.ProbeTimeout}.Resolved()
	c.ConnectTimeout, c.FirstByteTimeout, c.ProbeTimeout = wc.ConnectTimeout, wc.FirstByteTimeout, wc.RequestTimeout
	c.Workers = slices.Clone(c.Workers)
	for i := range c.Workers {
		if c.Workers[i].MaxInFlight <= 0 {
			c.Workers[i].MaxInFlight = 1
		}
	}
	return c
}

func (c Config) withDefaults() (Config, error) {
	c = c.Resolved()
	if len(c.Workers) == 0 {
		return c, errors.New("hub: no workers configured")
	}
	if c.UnknownModel != UnknownModelDefault && c.UnknownModel != UnknownModelError {
		return c, fmt.Errorf("hub: unknown_model must be %q or %q, got %q", UnknownModelDefault, UnknownModelError, c.UnknownModel)
	}
	seen := map[string]bool{}
	for _, w := range c.Workers {
		if seen[w.URL] {
			return c, fmt.Errorf("hub: worker %q listed twice", w.URL)
		}
		seen[w.URL] = true
	}
	return c, nil
}

// worker is one configured engine worker. Fields after client are guarded
// by Hub.mu.
type worker struct {
	url    string
	limit  int
	client *workerclient.Client
	kick   chan struct{}
	// probeMu serializes probes of this worker (poll loop, Refresh, kicks),
	// so a slow probe that started earlier cannot overwrite the result of
	// one that started later.
	probeMu sync.Mutex

	healthy  bool
	fails    int
	info     *workerclient.Info
	inFlight int
}

func (w *worker) serves(model string) bool {
	if !w.healthy || w.info == nil {
		return false
	}
	for _, m := range w.info.Models {
		if m.ID == model {
			return true
		}
	}
	return false
}

// Hub implements core.Service.
type Hub struct {
	cfg   Config
	authz core.Authorizer
	usage core.UsageRecorder
	log   *slog.Logger
	now   func() time.Time

	mu      sync.Mutex
	workers []*worker
	known   map[string]bool // every model id ever served, for the unknown-model policy
	pending []*job
	seq     uint64
	rr      int

	stop context.CancelFunc
	wg   sync.WaitGroup
}

var _ core.Service = (*Hub)(nil)

// New validates cfg and starts probing the workers in the background. It
// does not wait for any worker: startup must not fail while a worker is
// still loading. authz nil allows every model; usage nil records nothing;
// log nil uses slog.Default().
func New(cfg Config, authz core.Authorizer, usage core.UsageRecorder, log *slog.Logger) (*Hub, error) {
	h, err := newHub(cfg, authz, usage, log)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	h.stop = cancel
	for _, w := range h.workers {
		h.wg.Add(1)
		go func() {
			defer h.wg.Done()
			h.pollLoop(ctx, w)
		}()
	}
	h.log.Info("hub started", "workers", len(h.workers), "default_model", cfg.DefaultModel,
		"unknown_model", h.cfg.UnknownModel, "queue_length", h.cfg.QueueLength)
	return h, nil
}

func newHub(cfg Config, authz core.Authorizer, usage core.UsageRecorder, log *slog.Logger) (*Hub, error) {
	cfg, err := cfg.withDefaults()
	if err != nil {
		return nil, err
	}
	if log == nil {
		log = slog.Default()
	}
	h := &Hub{cfg: cfg, authz: authz, usage: usage, log: log, now: time.Now, known: map[string]bool{}, stop: func() {}}
	for _, wc := range cfg.Workers {
		c, err := workerclient.New(wc.URL, workerclient.Config{
			ConnectTimeout: cfg.ConnectTimeout, FirstByteTimeout: cfg.FirstByteTimeout, RequestTimeout: cfg.ProbeTimeout,
		}, log)
		if err != nil {
			return nil, fmt.Errorf("hub: %w", err)
		}
		h.workers = append(h.workers, &worker{url: c.URL(), limit: wc.MaxInFlight, client: c, kick: make(chan struct{}, 1)})
	}
	return h, nil
}

// Close stops the background probes. Jobs in flight are not interrupted;
// their callers own their contexts.
func (h *Hub) Close() {
	h.stop()
	h.wg.Wait()
	for _, w := range h.workers {
		w.client.Close()
	}
}

// Refresh probes every worker once, synchronously. Useful at startup to
// wait for workers, and in tests.
func (h *Hub) Refresh(ctx context.Context) {
	var wg sync.WaitGroup
	for _, w := range h.workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.probe(ctx, w)
		}()
	}
	wg.Wait()
}

// Ready reports whether the default model is served by a healthy worker,
// i.e. whether a request without a model name could run now.
func (h *Hub) Ready() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	def := h.defaultModelLocked()
	return def != "" && h.servedLocked(def)
}

// HealthHandler answers 200 {"status":"ok"} when Ready, else 503
// {"status":"unavailable"}. Both carry workers_up, workers_total and
// queued (jobs waiting for a worker).
func (h *Hub) HealthHandler() http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, _ *http.Request) {
		ready := h.Ready()
		h.mu.Lock()
		up := 0
		for _, w := range h.workers {
			if w.healthy {
				up++
			}
		}
		total, queued := len(h.workers), len(h.pending)
		h.mu.Unlock()
		status, code := "ok", http.StatusOK
		if !ready {
			status, code = "unavailable", http.StatusServiceUnavailable
		}
		rw.Header().Set("Content-Type", "application/json")
		rw.WriteHeader(code)
		_ = json.NewEncoder(rw).Encode(map[string]any{"status": status, "workers_up": up, "workers_total": total,
			"queued": queued})
	})
}

func (h *Hub) pollLoop(ctx context.Context, w *worker) {
	backoff := h.cfg.RetryMin
	for {
		wait := h.cfg.PollInterval
		if h.probe(ctx, w) {
			backoff = h.cfg.RetryMin
		} else {
			wait = backoff
			backoff = min(2*backoff, h.cfg.RetryMax)
		}
		t := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
		case <-w.kick:
			t.Stop()
		}
	}
}

// probe checks one worker and applies the result; it reports health.
func (h *Hub) probe(ctx context.Context, w *worker) bool {
	w.probeMu.Lock()
	defer w.probeMu.Unlock()
	err := w.client.Health(ctx)
	var info *workerclient.Info
	if err == nil {
		info, err = w.client.Info(ctx)
	}
	if err == nil && ctx.Err() != nil {
		err = ctx.Err()
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if err != nil {
		w.fails++
		h.log.Debug("worker probe failed", "worker", w.url, "fails", w.fails, "err", err)
		if w.healthy && w.fails >= h.cfg.FailThreshold {
			w.healthy = false
			h.log.Warn("worker down", "worker", w.url, "err", err)
			h.failUnservableLocked()
		}
		return false
	}
	w.fails = 0
	changed := !w.healthy || w.info == nil || !sameModels(w.info, info)
	w.healthy, w.info = true, info
	for _, m := range info.Models {
		h.known[m.ID] = true
	}
	if changed {
		ids := make([]string, 0, len(info.Models))
		for _, m := range info.Models {
			ids = append(ids, m.ID)
		}
		h.log.Info("worker up", "worker", w.url, "engine", info.Engine, "device", info.Device,
			"models", ids, "default", info.DefaultModel(), "version", info.Version)
		h.failUnservableLocked()
		h.dispatchLocked()
	} else {
		h.log.Debug("worker probe ok", "worker", w.url)
	}
	return true
}

func sameModels(a, b *workerclient.Info) bool {
	return a.Engine == b.Engine && a.Capabilities == b.Capabilities &&
		slices.Equal(a.Models, b.Models) && slices.Equal(a.Languages, b.Languages)
}

// kick asks for an immediate re-probe, used after a transport failure.
func (w *worker) kickProbe() {
	select {
	case w.kick <- struct{}{}:
	default:
	}
}
