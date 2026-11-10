// Package workertest provides an in-process fake engine worker speaking the
// Worker API, for tests of the hub and end-to-end setups.
//
// The fake is strict about the request shape (multipart part order, known
// option fields, even PCM length) so that callers drifting from the API
// fail loudly instead of passing against a lenient fake.
package workertest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Job is one request the fake received.
type Job struct {
	// Path is "/v1/transcribe" or "/v1/detect-language".
	Path    string
	Options Options
	// RawOptions is the options part exactly as sent.
	RawOptions []byte
	Audio      []byte
	// Model is Options.Model, or the info default when that was empty;
	// Engine is the info engine.
	Model, Engine string
	// ID is Options.JobID, the id pause requests name.
	ID string
	// Ctx is canceled when the client disconnects.
	Ctx context.Context

	w *Worker
}

// TranscribeFunc scripts the response to one /v1/transcribe job.
type TranscribeFunc func(job *Job, s *Stream)

// DetectFunc scripts the response to /v1/detect-language. Returning a
// non-empty code sends an HTTP error instead.
type DetectFunc func(job *Job) (language string, confidence float64, errCode string)

// Worker is a fake engine worker backed by httptest.Server.
type Worker struct {
	*httptest.Server

	mu          sync.Mutex
	info        Info
	healthy     bool
	transcribe  TranscribeFunc
	detect      DetectFunc
	jobs        []*Job
	inFlight    int
	maxInFlight int
	started     chan *Job

	// Open pause requests per job id; changed is closed and replaced on
	// every change so that waiting scripts wake up.
	holds       map[string]int
	changed     chan struct{}
	pauses      []PauseRequest
	acked       map[string]int
	refusePause bool
}

// DefaultInfo is a faster_whisper worker serving "tiny" with all
// capabilities except diarization.
func DefaultInfo() Info {
	return Info{
		Engine:    "faster_whisper",
		Version:   "fake",
		Device:    "cpu",
		Models:    []ModelInfo{{ID: "tiny", Default: true, Loaded: true}},
		Languages: core.Languages(),
		Capabilities: core.Capabilities{
			WordTimestamps: true, VADFilter: true, Diarize: false, Translate: true,
		},
	}
}

// New starts a healthy fake worker reporting info; it is closed by
// tb.Cleanup. Every received job is also sent to Started() when there is
// room in its buffer (64), so tests can wait for arrival without sleeping.
func New(tb testing.TB, info Info) *Worker {
	w := NewUnstarted(info)
	w.Start()
	tb.Cleanup(w.Close)
	return w
}

// NewUnstarted is New without starting the server or registering cleanup,
// so a standalone fake (cmd/fakeworker) can set its own Listener before
// calling Start.
func NewUnstarted(info Info) *Worker {
	w := &Worker{info: info, healthy: true, transcribe: Echo, started: make(chan *Job, 64),
		holds: map[string]int{}, changed: make(chan struct{}), acked: map[string]int{}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", w.handleHealth)
	mux.HandleFunc("GET /v1/info", w.handleInfo)
	mux.HandleFunc("POST /v1/transcribe", w.handleTranscribe)
	mux.HandleFunc("POST /v1/detect-language", w.handleDetect)
	mux.HandleFunc("POST /v1/pause", w.handlePause)
	w.Server = httptest.NewUnstartedServer(mux)
	return w
}

// SetHealthy makes /healthz and /v1/info answer 503 when false.
func (w *Worker) SetHealthy(ok bool) { w.mu.Lock(); w.healthy = ok; w.mu.Unlock() }

// SetInfo replaces the /v1/info body.
func (w *Worker) SetInfo(info Info) { w.mu.Lock(); w.info = info; w.mu.Unlock() }

// SetTranscribe replaces the transcription script; nil restores Echo.
func (w *Worker) SetTranscribe(f TranscribeFunc) {
	if f == nil {
		f = Echo
	}
	w.mu.Lock()
	w.transcribe = f
	w.mu.Unlock()
}

// SetDetect replaces the detect-language script.
func (w *Worker) SetDetect(f DetectFunc) { w.mu.Lock(); w.detect = f; w.mu.Unlock() }

// Started delivers jobs as they arrive.
func (w *Worker) Started() <-chan *Job { return w.started }

// Jobs returns every job received so far, in arrival order.
func (w *Worker) Jobs() []*Job {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]*Job(nil), w.jobs...)
}

// MaxInFlight returns the highest number of concurrently running jobs seen.
func (w *Worker) MaxInFlight() int { w.mu.Lock(); defer w.mu.Unlock(); return w.maxInFlight }

// SetPauseRefused makes POST /v1/pause answer 409 not_pausable, as a worker
// does whose engine cannot pause.
func (w *Worker) SetPauseRefused(refuse bool) { w.mu.Lock(); w.refusePause = refuse; w.mu.Unlock() }

// PauseRequests returns every pause request received so far, in order.
func (w *Worker) PauseRequests() []PauseRequest {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]PauseRequest(nil), w.pauses...)
}

// Paused returns how often the job with this id acknowledged a pause.
func (w *Worker) Paused(id string) int { w.mu.Lock(); defer w.mu.Unlock(); return w.acked[id] }

// handlePause holds a pause of the named job for as long as the client
// keeps the request open, like the real worker. Scripts honor it only in
// Stream.Await; a job may be paused before it arrives.
func (w *Worker) handlePause(rw http.ResponseWriter, r *http.Request) {
	var p PauseRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil || p.JobID == "" || p.Model == "" {
		writeError(rw, http.StatusBadRequest, "bad_request", fmt.Sprintf("pause request %+v: %v", p, err))
		return
	}
	w.mu.Lock()
	w.pauses = append(w.pauses, p)
	refuse := w.refusePause
	w.mu.Unlock()
	if refuse {
		writeError(rw, http.StatusConflict, "not_pausable", "this engine cannot pause")
		return
	}
	w.hold(p.JobID, 1)
	defer w.hold(p.JobID, -1)
	rw.Header().Set("Content-Type", "application/x-ndjson")
	rw.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(rw, `{"status":"pausing"}`+"\n")
	if f, ok := rw.(http.Flusher); ok {
		f.Flush()
	}
	<-r.Context().Done()
}

func (w *Worker) hold(id string, delta int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.holds[id] += delta
	if w.holds[id] <= 0 {
		delete(w.holds, id)
	}
	close(w.changed)
	w.changed = make(chan struct{})
}

func (w *Worker) holdState(id string) (held bool, changed <-chan struct{}) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.holds[id] > 0, w.changed
}

func (w *Worker) handleHealth(rw http.ResponseWriter, _ *http.Request) {
	w.mu.Lock()
	ok := w.healthy
	w.mu.Unlock()
	if !ok {
		writeError(rw, http.StatusServiceUnavailable, "unavailable", "loading")
		return
	}
	rw.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(rw, `{"status":"ok"}`)
}

func (w *Worker) handleInfo(rw http.ResponseWriter, _ *http.Request) {
	w.mu.Lock()
	ok, info := w.healthy, w.info
	w.mu.Unlock()
	if !ok {
		writeError(rw, http.StatusServiceUnavailable, "unavailable", "loading")
		return
	}
	rw.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(rw).Encode(info)
}

func (w *Worker) begin(job *Job) func() {
	w.mu.Lock()
	w.jobs = append(w.jobs, job)
	w.inFlight++
	w.maxInFlight = max(w.maxInFlight, w.inFlight)
	w.mu.Unlock()
	select {
	case w.started <- job:
	default:
	}
	return func() { w.mu.Lock(); w.inFlight--; w.mu.Unlock() }
}

func (w *Worker) handleTranscribe(rw http.ResponseWriter, r *http.Request) {
	job, err := parseJob(r)
	if err != nil {
		writeError(rw, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	w.mu.Lock()
	f, info := w.transcribe, w.info
	w.mu.Unlock()
	job.Model, job.Engine, job.ID, job.w = job.Options.Model, info.Engine, job.Options.JobID, w
	if job.Model == "" {
		job.Model = info.DefaultModel()
	}
	defer w.begin(job)()
	f(job, &Stream{rw: rw})
}

func (w *Worker) handleDetect(rw http.ResponseWriter, r *http.Request) {
	job, err := parseJob(r)
	if err != nil {
		writeError(rw, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	w.mu.Lock()
	f, info := w.detect, w.info
	w.mu.Unlock()
	job.Model, job.Engine = job.Options.Model, info.Engine
	if job.Model == "" {
		job.Model = info.DefaultModel()
	}
	defer w.begin(job)()
	lang, conf, code := "en", 0.98, ""
	if f != nil {
		lang, conf, code = f(job)
	}
	if code != "" {
		writeError(rw, statusFor(code), code, "scripted failure")
		return
	}
	model := job.Model
	rw.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(rw).Encode(map[string]any{
		"language": lang, "confidence": conf, "model": model, "engine": info.Engine,
	})
}

func parseJob(r *http.Request) (*Job, error) {
	mt, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mt != "multipart/form-data" {
		return nil, fmt.Errorf("want multipart/form-data, got %q", r.Header.Get("Content-Type"))
	}
	mr := multipart.NewReader(r.Body, params["boundary"])
	job := &Job{Path: r.URL.Path, Ctx: r.Context()}
	for i, name := range []string{"options", "audio"} {
		p, err := mr.NextPart()
		if err != nil {
			return nil, fmt.Errorf("part %d: %w", i, err)
		}
		if p.FormName() != name {
			return nil, fmt.Errorf("part %d is %q, want %q", i, p.FormName(), name)
		}
		data, err := io.ReadAll(p)
		if err != nil {
			return nil, err
		}
		if name == "options" {
			if ct := p.Header.Get("Content-Type"); ct != "application/json" {
				return nil, fmt.Errorf("options content type %q", ct)
			}
			job.RawOptions = data
			dec := json.NewDecoder(bytes.NewReader(data))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&job.Options); err != nil {
				return nil, fmt.Errorf("options: %w", err)
			}
		} else {
			job.Audio = data
		}
	}
	if _, err := mr.NextPart(); err != io.EOF {
		return nil, fmt.Errorf("unexpected extra part")
	}
	if len(job.Audio)%2 != 0 {
		return nil, fmt.Errorf("odd PCM length %d", len(job.Audio))
	}
	return job, nil
}

// Stream writes one NDJSON response.
type Stream struct {
	rw      http.ResponseWriter
	started bool
}

func (s *Stream) line(v any) {
	if !s.started {
		s.rw.Header().Set("Content-Type", "application/x-ndjson")
		s.rw.WriteHeader(http.StatusOK)
		s.started = true
	}
	b, _ := json.Marshal(v)
	_, _ = s.rw.Write(append(b, '\n'))
	if f, ok := s.rw.(http.Flusher); ok {
		f.Flush()
	}
}

// Info writes the info line.
func (s *Stream) Info(model, engine, language string, duration float64) {
	s.line(map[string]any{"type": "info", "model": model, "engine": engine, "language": language,
		"language_probability": 0.99, "duration": duration})
}

// Segment writes one segment line.
func (s *Stream) Segment(seg core.Segment) {
	s.line(map[string]any{"type": "segment", "segment": seg})
}

// Await waits until release delivers (a value or a close) while honoring
// pause requests the way the real worker does at a pause point: while a
// pause of this job is held it writes a paused line, waits for the hold to
// end, then writes a resumed line. It returns false when the client went
// away.
func (s *Stream) Await(job *Job, release <-chan struct{}) bool {
	for {
		held, changed := job.w.holdState(job.ID)
		if held && job.ID != "" {
			s.line(map[string]any{"type": "paused"})
			job.w.mu.Lock()
			job.w.acked[job.ID]++
			job.w.mu.Unlock()
			for held {
				select {
				case <-changed:
					held, changed = job.w.holdState(job.ID)
				case <-job.Ctx.Done():
					return false
				}
			}
			s.line(map[string]any{"type": "resumed"})
			continue
		}
		select {
		case <-release:
			return true
		case <-changed:
		case <-job.Ctx.Done():
			return false
		}
	}
}

// Pausing is Block that honors pause requests while it waits (see Await).
func Pausing(release <-chan struct{}) TranscribeFunc {
	return func(job *Job, s *Stream) {
		if s.Await(job, release) {
			Echo(job, s)
		}
	}
}

// Raw writes a literal line, for malformed or extended payloads.
func (s *Stream) Raw(line string) {
	if !s.started {
		s.line(json.RawMessage(`{"type":"noop"}`))
	}
	_, _ = io.WriteString(s.rw, strings.TrimRight(line, "\n")+"\n")
	if f, ok := s.rw.(http.Flusher); ok {
		f.Flush()
	}
}

// Done writes the final line; rendered is omitted when nil.
func (s *Stream) Done(text, language string, rendered *string) {
	m := map[string]any{"type": "done", "text": text, "language": language}
	if rendered != nil {
		m["rendered"] = *rendered
	}
	s.line(m)
}

// Error writes a final in-stream error line.
func (s *Stream) Error(code, message string) {
	s.line(map[string]any{"type": "error", "code": code, "message": message})
}

// HTTPError answers with an HTTP error instead of a stream; only valid
// before anything else was written.
func (s *Stream) HTTPError(code, message string) {
	writeError(s.rw, statusFor(code), code, message)
	s.started = true
}

// Echo is the default script: info, one segment per started second of
// audio, done. rendered is "txt:<text>" when legacy_output was set.
func Echo(job *Job, s *Stream) {
	secs := float64(len(job.Audio)/2) / core.SampleRate
	lang := "en"
	if job.Options.Language != nil {
		lang = *job.Options.Language
	}
	s.Info(job.Model, job.Engine, lang, secs)
	var text strings.Builder
	for i := 0; float64(i) < secs; i++ {
		seg := core.Segment{ID: i, Start: float64(i), End: min(float64(i+1), secs), Text: fmt.Sprintf(" s%d", i)}
		text.WriteString(seg.Text)
		s.Segment(seg)
	}
	var rendered *string
	if job.Options.LegacyOutput != nil {
		r := *job.Options.LegacyOutput + ":" + text.String()
		rendered = &r
	}
	s.Done(text.String(), lang, rendered)
}

// Block returns a script that waits until release is closed (or the client
// goes away) and then behaves like Echo.
func Block(release <-chan struct{}) TranscribeFunc {
	return func(job *Job, s *Stream) {
		select {
		case <-release:
		case <-job.Ctx.Done():
			return
		}
		Echo(job, s)
	}
}

func statusFor(code string) int {
	switch code {
	case "bad_request":
		return http.StatusBadRequest
	case "model_not_found":
		return http.StatusNotFound
	case "unavailable":
		return http.StatusServiceUnavailable
	case "not_pausable":
		return http.StatusConflict
	default:
		return http.StatusInternalServerError
	}
}

func writeError(rw http.ResponseWriter, status int, code, message string) {
	rw.Header().Set("Content-Type", "application/json")
	rw.WriteHeader(status)
	_ = json.NewEncoder(rw).Encode(map[string]any{"error": map[string]string{"code": code, "message": message}})
}
