package openaiapi

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

type deltaEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
}

type doneEvent struct {
	Type  string `json:"type"`
	Text  string `json:"text"`
	Usage *usage `json:"usage,omitempty"`
}

// sseWriter starts the event stream lazily, so a failure before the first
// segment still gets a regular HTTP error status.
type sseWriter struct {
	w       http.ResponseWriter
	rc      *http.ResponseController
	started bool
	log     *slog.Logger
}

func (e *sseWriter) send(v any) error {
	if !e.started {
		h := e.w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		// Reverse proxies (nginx) buffer otherwise and defeat streaming.
		h.Set("X-Accel-Buffering", "no")
		e.w.WriteHeader(http.StatusOK)
		e.started = true
	}
	// Only our own event structs get here; they always marshal.
	b, _ := json.Marshal(v)
	e.log.Debug("openai: sse event", "data", string(b))
	if _, err := e.w.Write([]byte("data: " + string(b) + "\n\n")); err != nil {
		return err
	}
	return e.rc.Flush()
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request, req *core.Request, log *slog.Logger, start time.Time) {
	ev := &sseWriter{w: w, rc: http.NewResponseController(w), log: log}
	first := true
	// A write error (client gone) is returned from onSegment, which cancels
	// the job; r.Context() is canceled on disconnect as well.
	onSegment := func(seg core.Segment) error {
		text := seg.Text
		if first {
			text = strings.TrimLeft(text, " ")
		}
		if text == "" {
			return nil
		}
		first = false
		return ev.send(deltaEvent{Type: "transcript.text.delta", Delta: text})
	}
	res, err := s.svc.Transcribe(r.Context(), req, onSegment)
	if err != nil {
		if !ev.started {
			s.fail(w, log, err, start)
			return
		}
		ae := toAPIError(err)
		log.Debug("openai: stream failed", "status", ae.Status, "error", err.Error(),
			"elapsed", time.Since(start).Round(time.Millisecond))
		if core.KindOf(err) == core.KindCanceled {
			return
		}
		// The SDK raises APIError for a data event carrying "error".
		_ = ev.send(errorBody{Error: ae})
		return
	}
	dur := res.Duration
	if dur == 0 {
		dur = req.Audio.Seconds()
	}
	if err := ev.send(doneEvent{Type: "transcript.text.done", Text: fullText(res), Usage: durationUsage(dur)}); err != nil {
		log.Debug("openai: stream client gone before done", "error", err.Error())
		return
	}
	log.Debug("openai: job done", "model", res.Model, "format", "stream", "audio_seconds", dur,
		"segments", len(res.Segments), "elapsed", time.Since(start).Round(time.Millisecond))
}
