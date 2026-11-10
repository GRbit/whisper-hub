package hub

import (
	"context"
	"log/slog"
	"math"
	"slices"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/hub/internal/workerclient"
)

// detectMaxBytes is 30 s of PCM: every engine only looks at the first 30 s
// for language detection, as upstream does, so the rest is not sent.
const detectMaxBytes = 30 * core.SampleRate * 2

// Transcribe implements core.Service.
func (h *Hub) Transcribe(ctx context.Context, req *core.Request, onSegment func(core.Segment) error) (*core.Result, error) {
	start := h.now()
	task := req.Task
	if task == "" {
		task = core.TaskTranscribe
	}
	ev := core.UsageEvent{
		Time: start, User: core.IdentityFrom(ctx).User, Protocol: req.Protocol, Endpoint: req.Endpoint,
		Model: req.Model, Task: task, AudioSeconds: req.Audio.Seconds(),
	}
	h.log.Debug("transcribe request", "user", ev.User, "protocol", req.Protocol, "endpoint", req.Endpoint,
		"model", req.Model, "task", task, "language", req.Language, "audio_s", ev.AudioSeconds,
		"priority", h.priorityOf(ctx, req.Protocol), "vad", req.VADFilter, "words", req.WordTimestamps, "diarize", req.Diarize,
		"legacy_output", req.LegacyOutput)
	res, err := h.transcribe(ctx, req, onSegment, &ev)
	h.finish(ctx, &ev, start, err)
	return res, err
}

func (h *Hub) transcribe(ctx context.Context, req *core.Request, onSegment func(core.Segment) error, ev *core.UsageEvent) (*core.Result, error) {
	h.mu.Lock()
	view, err := h.resolveLocked(ctx, req.Model)
	h.mu.Unlock()
	if err != nil {
		return nil, err
	}
	ev.Model, ev.Engine = view.id, view.engine
	opts, err := h.prepare(req, view)
	if err != nil {
		return nil, err
	}
	ev.Task = opts.Task
	j := &job{ctx: ctx, model: view.id, prio: h.priorityOf(ctx, req.Protocol), pausable: true,
		maxRestarts: *h.cfg.MaxRestarts}
	seg := onSegment
	if onSegment != nil {
		seg = func(s core.Segment) error {
			if !h.delivering(j) {
				return core.WrapError(core.KindCanceled, errRestart, "job canceled for a restart")
			}
			return onSegment(s)
		}
	}
	var res *core.Result
	err = h.schedule(j, func(actx context.Context, w *worker, id string, onPause func(bool)) error {
		o := opts
		o.JobID = id
		var err error
		res, err = w.client.Transcribe(actx, o, req.Audio, seg, onPause)
		if err != nil && core.KindOf(err) == core.KindUnavailable {
			w.kickProbe()
		}
		return err
	})
	ev.Queued = h.queuedOf(j)
	if err != nil {
		return nil, err
	}
	if res.Model == "" {
		res.Model = view.id
	}
	if res.Engine == "" {
		res.Engine = view.engine
	}
	if res.Duration == 0 {
		res.Duration = req.Audio.Seconds()
	}
	ev.Engine = res.Engine
	return res, nil
}

// prepare validates req against what the model's engine can do and builds
// the worker options.
//
// Legacy (/asr) requests never fail on an engine capability: upstream
// whisper-asr-webservice silently ignores vad_filter and word_timestamps on
// engines without them, ignores diarize without whisperx and an HF token,
// and passes translate to English-only models, which then transcribe. So
// for legacy the unsupported option is dropped (translate becomes
// transcribe). Other protocols get KindBadRequest for options that change
// the shape of the answer (translate, word timestamps, diarize); vad_filter
// is only a hint and is dropped for everyone.
func (h *Hub) prepare(req *core.Request, v modelView) (workerclient.Options, error) {
	legacy := req.Protocol == core.ProtocolLegacy
	switch req.Task {
	case "", core.TaskTranscribe, core.TaskTranslate:
	default:
		return workerclient.Options{}, core.Errorf(core.KindBadRequest, "unsupported task %q", req.Task)
	}
	switch req.LegacyOutput {
	case "", "txt", "vtt", "srt", "tsv", "json":
	default:
		return workerclient.Options{}, core.Errorf(core.KindBadRequest, "unsupported output format %q", req.LegacyOutput)
	}
	if len(req.Audio) == 0 {
		return workerclient.Options{}, core.Errorf(core.KindBadAudio, "audio is empty")
	}
	opts := workerclient.OptionsFromRequest(v.id, req)
	drop := func(what string) {
		h.log.Debug("ignoring unsupported option", "option", what, "model", v.id, "engine", v.engine)
	}
	bad := func(format string, args ...any) (workerclient.Options, error) {
		return workerclient.Options{}, core.Errorf(core.KindBadRequest, format, args...)
	}

	if opts.Task == core.TaskTranslate && !v.caps.Translate {
		if !legacy {
			return bad("model %q does not support translation", v.id)
		}
		drop("translate")
		opts.Task = core.TaskTranscribe
	}
	if opts.WordTimestamps && !v.caps.WordTimestamps {
		if !legacy {
			return bad("model %q does not support word timestamps", v.id)
		}
		drop("word_timestamps")
		opts.WordTimestamps = false
	}
	if opts.VADFilter && !v.caps.VADFilter {
		drop("vad_filter")
		opts.VADFilter = false
	}
	if opts.Diarize && !v.caps.Diarize {
		if !legacy {
			return bad("model %q does not support diarization", v.id)
		}
		drop("diarize")
		opts.Diarize = false
	}
	if !opts.Diarize {
		opts.MinSpeakers, opts.MaxSpeakers = nil, nil
	} else if !legacy {
		mn, mx := opts.MinSpeakers, opts.MaxSpeakers
		if (mn != nil && *mn < 1) || (mx != nil && *mx < 1) || (mn != nil && mx != nil && *mn > *mx) {
			return bad("invalid speaker count range")
		}
	}
	if !legacy {
		// Legacy validates language and temperature against its own enum
		// upstream-style; an engine-specific check here could only add errors.
		if req.Language != "" {
			if core.LanguageName(req.Language) == "" {
				return bad("unsupported language %q", req.Language)
			}
			if len(v.languages) > 0 && !slices.Contains(v.languages, req.Language) {
				return bad("model %q does not support language %q", v.id, req.Language)
			}
		}
		if t := req.Temperature; t != nil && (math.IsNaN(*t) || *t < 0 || *t > 1) {
			return bad("temperature must be between 0 and 1")
		}
	}
	return opts, nil
}

// DetectLanguage implements core.Service.
func (h *Hub) DetectLanguage(ctx context.Context, req *core.DetectRequest) (*core.Detection, error) {
	start := h.now()
	audio := req.Audio
	if len(audio) > detectMaxBytes {
		audio = audio[:detectMaxBytes]
	}
	ev := core.UsageEvent{
		Time: start, User: core.IdentityFrom(ctx).User, Protocol: req.Protocol, Endpoint: req.Endpoint,
		Model: req.Model, AudioSeconds: audio.Seconds(),
	}
	h.log.Debug("detect-language request", "user", ev.User, "protocol", req.Protocol, "endpoint", req.Endpoint,
		"model", req.Model, "audio_s", req.Audio.Seconds(), "priority", h.priorityOf(ctx, req.Protocol))
	det, err := h.detect(ctx, req, audio, &ev)
	h.finish(ctx, &ev, start, err)
	return det, err
}

func (h *Hub) detect(ctx context.Context, req *core.DetectRequest, audio core.PCM, ev *core.UsageEvent) (*core.Detection, error) {
	if len(audio) == 0 {
		return nil, core.Errorf(core.KindBadAudio, "audio is empty")
	}
	h.mu.Lock()
	view, err := h.resolveLocked(ctx, req.Model)
	h.mu.Unlock()
	if err != nil {
		return nil, err
	}
	ev.Model, ev.Engine = view.id, view.engine
	j := &job{ctx: ctx, model: view.id, prio: h.priorityOf(ctx, req.Protocol), maxRestarts: *h.cfg.MaxRestarts}
	var det *core.Detection
	err = h.schedule(j, func(actx context.Context, w *worker, _ string, _ func(bool)) error {
		var err error
		det, err = w.client.DetectLanguage(actx, view.id, audio)
		if err != nil && core.KindOf(err) == core.KindUnavailable {
			w.kickProbe()
		}
		return err
	})
	ev.Queued = h.queuedOf(j)
	if err != nil {
		return nil, err
	}
	if det.Model == "" {
		det.Model = view.id
	}
	if det.Engine == "" {
		det.Engine = view.engine
	}
	return det, nil
}

// queuedOf is the time j spent waiting for a slot or paused.
func (h *Hub) queuedOf(j *job) time.Duration {
	h.mu.Lock()
	defer h.mu.Unlock()
	return j.queued
}

// finish records usage and writes the one info line per job.
func (h *Hub) finish(ctx context.Context, ev *core.UsageEvent, start time.Time, err error) {
	ev.Elapsed = h.now().Sub(start)
	ev.Status = "ok"
	if err != nil {
		ev.Status = core.KindOf(err).String()
		ev.Error = core.ClientMessage(err)
	}
	attrs := []any{"user", ev.User, "protocol", ev.Protocol, "endpoint", ev.Endpoint, "model", ev.Model,
		"engine", ev.Engine, "audio_s", math.Round(ev.AudioSeconds*100) / 100,
		"queued_ms", ev.Queued.Milliseconds(), "elapsed_ms", ev.Elapsed.Milliseconds(), "status", ev.Status}
	// The usage recorder writes the one Info line per job; repeating it here
	// would double every job in the log. Without a recorder this is it.
	level := slog.LevelDebug
	if h.usage == nil {
		level = slog.LevelInfo
	}
	switch kind := core.KindOf(err); {
	case err == nil:
		h.log.Log(ctx, level, "job done", attrs...)
	case kind == core.KindInternal || kind == core.KindUnavailable:
		// Operator-relevant, and the cause (never shown to clients) is only here.
		h.log.Warn("job failed", append(attrs, "err", err.Error())...)
	default:
		h.log.Log(ctx, level, "job failed", append(attrs, "err", err.Error())...)
	}
	if h.usage != nil {
		// Detached so a canceled request still gets its record.
		h.usage.Record(context.WithoutCancel(ctx), *ev)
	}
}
