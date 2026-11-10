// Package openaiapi serves the OpenAI-compatible speech-to-text API
// (/v1/audio/transcriptions, /v1/audio/translations, /v1/models) on top of
// core.Service, at the level of the whisper-1 model.
package openaiapi

import (
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Endpoint paths, also used as core.Request.Endpoint.
const (
	pathTranscriptions = "/v1/audio/transcriptions"
	pathTranslations   = "/v1/audio/translations"
	pathModels         = "/v1/models"
)

// Config configures the OpenAI frontend.
type Config struct {
	// MaxUploadBytes caps the request body size; larger uploads get 413.
	// Default 0 means unlimited: unlike OpenAI's 25 MB, self-hosted users
	// send hour-long recordings.
	MaxUploadBytes int64 `yaml:"max_upload_bytes"`
	// OwnedBy is the owned_by value reported by /v1/models. Default
	// "whisper-hub".
	OwnedBy string `yaml:"owned_by"`
}

// Server is the OpenAI API frontend.
type Server struct {
	cfg     Config
	svc     core.Service
	dec     core.AudioDecoder
	log     *slog.Logger
	created int64
}

// New returns a Server. A nil log uses slog.Default().
func New(cfg Config, svc core.Service, dec core.AudioDecoder, log *slog.Logger) *Server {
	cfg = cfg.Resolved()
	if log == nil {
		log = slog.Default()
	}
	return &Server{
		cfg: cfg, svc: svc, dec: dec,
		log: log.With("component", "openaiapi"),
		// Models carry no creation time; process start is stable enough for
		// clients that sort by it.
		created: time.Now().Unix(),
	}
}

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it.
func (c Config) Resolved() Config {
	if c.OwnedBy == "" {
		c.OwnedBy = "whisper-hub"
	}
	return c
}

// Register adds the OpenAI routes to mux.
//
// It also claims the "/v1/" subtree so that unknown paths and wrong methods
// get OpenAI-shaped 404s instead of the mux's plain-text replies.
func (s *Server) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST "+pathTranscriptions, func(w http.ResponseWriter, r *http.Request) {
		s.handleAudio(w, r, core.TaskTranscribe, pathTranscriptions)
	})
	mux.HandleFunc("POST "+pathTranslations, func(w http.ResponseWriter, r *http.Request) {
		s.handleAudio(w, r, core.TaskTranslate, pathTranslations)
	})
	mux.HandleFunc("GET "+pathModels, s.handleModels)
	mux.HandleFunc("GET "+pathModels+"/{id...}", s.handleModel)
	mux.HandleFunc("/v1/", s.handleNotFound)
}

func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	s.log.Debug("openai: unknown route", "method", r.Method, "path", r.URL.Path)
	writeJSON(w, http.StatusNotFound, errorBody{Error: &apiError{
		Message: fmt.Sprintf("Invalid URL (%s %s)", r.Method, r.URL.Path),
		Type:    typeInvalidRequest,
	}})
}

func (s *Server) handleAudio(w http.ResponseWriter, r *http.Request, task core.Task, endpoint string) {
	start := time.Now()
	ctx := r.Context()
	id := core.IdentityFrom(ctx)
	log := s.log.With("endpoint", endpoint, "user", id.User)
	log.Debug("openai: request", "remote", r.RemoteAddr, "content_length", r.ContentLength)

	p, err := s.parseForm(ctx, w, r, task, log)
	if err != nil {
		s.fail(w, log, err, start)
		return
	}
	log.Debug("openai: parameters", "model", p.model, "language", p.language, "languages", p.languages,
		"prompt_len", len(p.prompt), "temperature", p.temperature, "response_format", p.format,
		"granularity_segment", p.granSegment, "granularity_word", p.granWord, "stream", p.stream)

	req := &core.Request{
		Audio:       p.audio,
		Model:       p.model,
		Task:        task,
		Language:    p.language,
		Prompt:      p.prompt,
		Temperature: p.temperature,
		// Only verbose_json can carry words; asking otherwise costs latency.
		WordTimestamps: p.granWord && p.format == formatVerboseJSON && task == core.TaskTranscribe,
		Protocol:       core.ProtocolOpenAI,
		Endpoint:       endpoint,
	}

	// Like the real API for whisper-1, stream is ignored for formats that
	// cannot be expressed as text deltas.
	if p.stream && task == core.TaskTranscribe && (p.format == formatJSON || p.format == formatText) {
		s.stream(w, r, req, log, start)
		return
	}

	res, err := s.svc.Transcribe(ctx, req, nil)
	if err != nil {
		s.fail(w, log, err, start)
		return
	}
	audioSec := p.audio.Seconds()
	if res.Duration == 0 {
		res.Duration = audioSec
	}
	switch p.format {
	case formatText:
		writeText(w, fullText(res)+"\n")
	case formatSRT:
		writeText(w, renderSRT(res.Segments))
	case formatVTT:
		writeText(w, renderVTT(res.Segments))
	case formatVerboseJSON:
		writeJSON(w, http.StatusOK, buildVerbose(res, p, task, audioSec))
	default:
		out := jsonResponse{Text: fullText(res)}
		if task == core.TaskTranscribe {
			out.Usage = durationUsage(res.Duration)
		}
		writeJSON(w, http.StatusOK, out)
	}
	log.Debug("openai: job done", "model", res.Model, "format", p.format, "audio_seconds", res.Duration,
		"segments", len(res.Segments), "elapsed", time.Since(start).Round(time.Millisecond))
}

func (s *Server) fail(w http.ResponseWriter, log *slog.Logger, err error, start time.Time) {
	ae := toAPIError(err)
	attrs := []any{"status", ae.Status, "error", err.Error(), "elapsed", time.Since(start).Round(time.Millisecond)}
	if ae.Status >= 500 && ae.Status != 499 {
		log.Error("openai: job failed", attrs...)
	} else {
		log.Debug("openai: job failed", attrs...)
	}
	WriteError(w, ae)
}

func writeText(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(body))
}

type modelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type modelList struct {
	Object string        `json:"object"`
	Data   []modelObject `json:"data"`
}

// modelNames lists every name the caller may use: real ids first, then
// aliases, each once, because OpenAI clients look models up by alias.
func modelNames(models []core.Model) []string {
	seen := map[string]bool{}
	var names []string
	add := func(n string) {
		if n != "" && !seen[n] {
			seen[n] = true
			names = append(names, n)
		}
	}
	for _, m := range models {
		add(m.ID)
	}
	for _, m := range models {
		for _, a := range m.Aliases {
			add(a)
		}
	}
	return names
}

func (s *Server) modelObj(name string) modelObject {
	return modelObject{ID: name, Object: "model", Created: s.created, OwnedBy: s.cfg.OwnedBy}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	names := modelNames(s.svc.Models(r.Context()))
	s.log.Debug("openai: list models", "user", core.IdentityFrom(r.Context()).User, "count", len(names))
	out := modelList{Object: "list", Data: make([]modelObject, 0, len(names))}
	for _, n := range names {
		out.Data = append(out.Data, s.modelObj(n))
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.log.Debug("openai: retrieve model", "user", core.IdentityFrom(r.Context()).User, "model", id)
	for _, n := range modelNames(s.svc.Models(r.Context())) {
		if n == id {
			writeJSON(w, http.StatusOK, s.modelObj(n))
			return
		}
	}
	WriteError(w, &apiError{
		Status:  http.StatusNotFound,
		Message: fmt.Sprintf("The model `%s` does not exist or you do not have access to it.", id),
		Type:    typeInvalidRequest,
		Param:   strPtr("model"),
		Code:    strPtr("model_not_found"),
	})
}
