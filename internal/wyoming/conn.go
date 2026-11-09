package wyoming

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// maxQueued bounds finished utterances waiting behind a running one on the
// same connection; each can hold up to MaxUtteranceBytes.
const maxQueued = 2

// Wyoming's defaults when a client leaves the audio format out; HA always
// sends 16 kHz, 16-bit mono.
const (
	defaultRate     = 16000
	defaultWidth    = 2
	defaultChannels = 1
)

const (
	msgShutdown = "server is shutting down"
	msgIdle     = "idle timeout"
)

type conn struct {
	s      *Server
	nc     net.Conn
	log    *slog.Logger
	ctx    context.Context // canceled on disconnect, protocol error or hard shutdown
	cancel context.CancelFunc
	hard   context.Context

	wmu       sync.Mutex
	w         *frameWriter
	closeOnce sync.Once

	// Main loop state, touched only by run and its helpers.
	next       request // settings from the last transcribe, for the next stream
	stream     *utterance
	queue      []*utterance
	jobRunning bool
	seq        int

	jobs   sync.WaitGroup
	jobRes chan bool // true: keep the connection
}

type request struct {
	model    string
	language string
	set      bool
}

type format struct{ rate, width, channels int }

func (f format) bytesPerSecond() int64 { return int64(f.rate) * int64(f.width) * int64(f.channels) }

type segment struct {
	format
	data []byte
}

type utterance struct {
	id      int
	req     request
	cur     format
	segs    []segment
	bytes   int64
	seconds float64
	chunks  int
	started time.Time
}

func (u *utterance) add(f format, data []byte) {
	if n := len(u.segs); n > 0 && u.segs[n-1].format == f {
		u.segs[n-1].data = append(u.segs[n-1].data, data...)
	} else {
		u.segs = append(u.segs, segment{format: f, data: append([]byte(nil), data...)})
	}
	u.bytes += int64(len(data))
	u.seconds += float64(len(data)) / float64(f.bytesPerSecond())
	u.chunks++
}

// run is the connection's event loop. It returns why the connection ends.
func (c *conn) run(shutdown context.Context) string {
	events := make(chan *event)
	readErr := make(chan error, 1)
	go func() {
		r := newFrameReader(c.nc, c.s.limits)
		for {
			ev, err := r.ReadEvent()
			if err != nil {
				readErr <- err
				return
			}
			select {
			case events <- ev:
			case <-c.ctx.Done():
				return
			}
		}
	}()

	var idle <-chan time.Time
	var idleTimer *time.Timer
	if enabled(c.s.cfg.IdleTimeout) {
		idleTimer = time.NewTimer(c.s.cfg.IdleTimeout)
		defer idleTimer.Stop()
		idle = idleTimer.C
	}
	resetIdle := func() {
		if idleTimer != nil {
			idleTimer.Reset(c.s.cfg.IdleTimeout)
		}
	}

	shutdownC := shutdown.Done()
	shuttingDown := false
	readClosed := false
	busy := func() bool { return c.jobRunning || len(c.queue) > 0 }

	for {
		select {
		case ev := <-events:
			resetIdle()
			if shuttingDown && (ev.Type == typeAudioStart || ev.Type == typeAudioChunk) {
				c.sendError(core.KindUnavailable.String(), msgShutdown)
				return "shutdown"
			}
			if reason := c.handleEvent(ev); reason != "" {
				return reason
			}
		case err := <-readErr:
			if errors.Is(err, io.EOF) && c.stream == nil && busy() {
				// EOF after audio-stop may be a half-close by a client still
				// reading (nc, scripts). TCP cannot tell that from a full
				// close, and a gone client only costs one short interactive
				// job, so let pending utterances finish.
				c.log.Debug("wyoming client closed its side, finishing pending transcriptions", "queued", len(c.queue))
				readErr, readClosed = nil, true
				continue
			}
			return c.readFailed(err)
		case ok := <-c.jobRes:
			c.jobRunning = false
			if !ok {
				return "transcription failed"
			}
			c.startNext()
			if shuttingDown && !busy() {
				return "shutdown"
			}
			if readClosed && !busy() {
				return "client closed"
			}
		case <-idle:
			if busy() {
				resetIdle()
				continue
			}
			c.log.Debug("wyoming idle timeout", "timeout", c.s.cfg.IdleTimeout, "recording", c.stream != nil)
			if c.stream != nil {
				c.sendError(core.KindBadRequest.String(), msgIdle)
			}
			return "idle"
		case <-shutdownC:
			shutdownC = nil
			shuttingDown = true
			if c.stream != nil {
				c.sendError(core.KindUnavailable.String(), msgShutdown)
				return "shutdown"
			}
			if !busy() {
				return "shutdown"
			}
			c.log.Debug("wyoming shutdown: waiting for pending transcriptions", "queued", len(c.queue))
		case <-c.ctx.Done():
			// Hard shutdown or a failed write. A running job sees the same
			// cancellation and writes the shutdown error; let it, before the
			// caller closes the socket.
			if c.jobRunning {
				<-c.jobRes
				c.jobRunning = false
			}
			return "canceled"
		}
	}
}

func (c *conn) readFailed(err error) string {
	var fe *frameError
	switch {
	case errors.Is(err, io.EOF):
		if c.stream != nil {
			c.log.Debug("wyoming client disconnected mid-utterance, dropping it", "utterance", c.stream.id)
		}
		return "client closed"
	case errors.Is(err, io.ErrUnexpectedEOF):
		c.log.Debug("wyoming client disconnected inside a frame")
		return "client closed"
	case errors.As(err, &fe):
		c.log.Info("wyoming protocol error", "err", fe.Msg)
		c.sendError(core.KindBadRequest.String(), fe.Error())
		return "protocol error"
	default:
		if c.ctx.Err() == nil {
			c.log.Debug("wyoming read failed", "err", err)
		}
		return "read error"
	}
}

// handleEvent processes one client event; a non-empty result closes the
// connection with that reason.
func (c *conn) handleEvent(ev *event) string {
	if ev.Type != typeAudioChunk {
		c.log.Debug("wyoming event received", "type", ev.Type, "payload_bytes", len(ev.Payload))
	}
	switch ev.Type {
	case typeDescribe:
		c.write(typeInfo, c.info(), nil)
		c.log.Debug("wyoming info sent")
	case typePing:
		var p pingData
		_ = ev.decodeData(&p)
		c.write(typePong, p, nil)
	case typeSelectProgram:
		var p selectProgramData
		_ = ev.decodeData(&p)
		if p.Name != c.s.cfg.ProgramName {
			c.log.Debug("wyoming select-program for unknown program, keeping ours", "name", p.Name)
		}
	case typeTranscribe:
		var t transcribeData
		if err := ev.decodeData(&t); err != nil {
			return c.badRequest(err)
		}
		req := request{set: true}
		if t.Name != nil {
			req.model = *t.Name
		}
		if t.Language != nil {
			req.language = *t.Language
		}
		c.log.Debug("wyoming transcribe", "model", req.model, "language", req.language,
			"vad_sensitivity", deref(t.VADSensitivity), "names", len(t.TranscriptNames), "terms", len(t.TranscriptTerms))
		if c.stream != nil {
			// Audio is only sent to the engine at audio-stop, so a late
			// transcribe can still apply to the stream being recorded.
			c.stream.req = req
		} else {
			c.next = req
		}
	case typeAudioStart:
		var af audioFormat
		if err := ev.decodeData(&af); err != nil {
			return c.badRequest(err)
		}
		f, err := resolveFormat(af, format{defaultRate, defaultWidth, defaultChannels})
		if err != nil {
			return c.badAudio(err.Error())
		}
		if c.stream != nil {
			c.log.Debug("wyoming audio-start before audio-stop, dropping previous stream",
				"utterance", c.stream.id, "bytes", c.stream.bytes)
		}
		c.beginStream(f)
		c.log.Debug("wyoming audio started", "utterance", c.stream.id,
			"rate", f.rate, "width", f.width, "channels", f.channels)
	case typeAudioChunk:
		return c.audioChunk(ev)
	case typeAudioStop:
		u := c.stream
		c.stream = nil
		if u == nil {
			// Nothing recorded: answer anyway so the client is not left waiting.
			u = c.newUtterance(format{defaultRate, defaultWidth, defaultChannels})
		}
		c.log.Debug("wyoming audio stopped", "utterance", u.id, "chunks", u.chunks,
			"bytes", u.bytes, "seconds", round2(u.seconds))
		if len(c.queue) >= maxQueued {
			c.sendError(core.KindUnavailable.String(), "too many pending utterances on this connection")
			return "queue full"
		}
		c.queue = append(c.queue, u)
		c.startNext()
	default:
		c.log.Debug("wyoming event ignored", "type", ev.Type)
	}
	return ""
}

func (c *conn) newUtterance(f format) *utterance {
	c.seq++
	u := &utterance{id: c.seq, req: c.next, cur: f, started: time.Now()}
	c.next = request{}
	return u
}

func (c *conn) beginStream(f format) { c.stream = c.newUtterance(f) }

func (c *conn) audioChunk(ev *event) string {
	var af audioFormat
	if err := ev.decodeData(&af); err != nil {
		return c.badRequest(err)
	}
	if c.stream == nil {
		// Same leniency as wyoming-faster-whisper: audio without audio-start
		// starts a stream in the chunk's own format.
		f, err := resolveFormat(af, format{defaultRate, defaultWidth, defaultChannels})
		if err != nil {
			return c.badAudio(err.Error())
		}
		c.beginStream(f)
		c.log.Debug("wyoming audio-chunk without audio-start, stream started", "utterance", c.stream.id)
	}
	u := c.stream
	f, err := resolveFormat(af, u.cur)
	if err != nil {
		return c.badAudio(err.Error())
	}
	u.cur = f
	if len(ev.Payload) == 0 {
		return ""
	}
	if u.chunks == 0 {
		c.log.Debug("wyoming first audio chunk", "utterance", u.id, "bytes", len(ev.Payload))
	}
	u.add(f, ev.Payload)
	if u.bytes > c.s.cfg.MaxUtteranceBytes ||
		(enabled(c.s.cfg.MaxUtteranceDuration) && u.seconds > c.s.cfg.MaxUtteranceDuration.Seconds()) {
		c.log.Info("wyoming utterance too long", "utterance", u.id, "bytes", u.bytes, "seconds", round2(u.seconds))
		c.sendError(core.KindBadRequest.String(), "utterance exceeds the audio limit")
		return "utterance too long"
	}
	return ""
}

func resolveFormat(af audioFormat, base format) (format, error) {
	f := base
	if af.Rate != nil {
		f.rate = *af.Rate
	}
	if af.Width != nil {
		f.width = *af.Width
	}
	if af.Channels != nil {
		f.channels = *af.Channels
	}
	switch {
	case f.rate < 1000 || f.rate > 384000:
		return f, errors.New("unsupported sample rate")
	case f.width < 1 || f.width > 4:
		return f, errors.New("unsupported sample width")
	case f.channels < 1 || f.channels > 32:
		return f, errors.New("unsupported channel count")
	}
	return f, nil
}

func (c *conn) badRequest(err error) string {
	c.log.Debug("wyoming bad event data", "err", err)
	c.sendError(core.KindBadRequest.String(), "invalid event data")
	return "bad request"
}

func (c *conn) badAudio(msg string) string {
	c.log.Debug("wyoming bad audio format", "err", msg)
	c.sendError(core.KindBadAudio.String(), msg)
	return "bad audio"
}

func (c *conn) startNext() {
	if c.jobRunning || len(c.queue) == 0 {
		return
	}
	u := c.queue[0]
	c.queue = c.queue[1:]
	c.jobRunning = true
	c.jobs.Add(1)
	go func() {
		defer c.jobs.Done()
		c.jobRes <- c.transcribe(u)
	}()
}

// transcribe runs one utterance and writes its result; false closes the
// connection, which is how every Home Assistant version notices a failure
// (only recent ones read the error event).
func (c *conn) transcribe(u *utterance) bool {
	streaming := c.s.cfg.StreamTranscripts
	start := time.Now()
	log := c.log.With("utterance", u.id)

	if u.bytes == 0 {
		log.Debug("wyoming utterance without audio, empty transcript")
		return c.writeResult(streaming, false, "", "", "")
	}

	pcm, err := c.convert(u)
	if err != nil {
		return c.jobFailed(log, u, start, "", err)
	}

	models := c.s.svc.Models(c.ctx)
	m, _ := modelFor(models, u.req.model)
	lang := resolveLanguage(u.req.language, m.Languages)
	if u.req.language != "" && !strings.EqualFold(lang, u.req.language) {
		log.Debug("wyoming language mapped", "requested", u.req.language, "using", orAuto(lang), "model", m.ID)
	}

	req := &core.Request{
		Audio:    pcm,
		Model:    u.req.model,
		Task:     core.TaskTranscribe,
		Language: lang,
		Priority: core.PriorityInteractive,
		Protocol: core.ProtocolWyoming,
		Endpoint: typeTranscribe,
	}

	var onSegment func(core.Segment) error
	started := false
	var sofar strings.Builder
	lastChunk := ""
	if streaming {
		onSegment = func(seg core.Segment) error {
			if !started {
				if err := c.write(typeTranscriptStart, transcriptStartData{Language: lang}, nil); err != nil {
					return err
				}
				started = true
			}
			sofar.WriteString(seg.Text)
			lastChunk = strings.TrimSpace(sofar.String())
			log.Debug("wyoming transcript chunk", "segment", seg.ID, "chars", len(lastChunk))
			return c.write(typeTranscriptChunk, transcriptChunkData{Text: lastChunk}, nil)
		}
	}

	log.Debug("wyoming transcription submitted", "model", req.Model, "language", orAuto(lang),
		"audio_seconds", round2(pcm.Seconds()))
	res, err := c.s.svc.Transcribe(c.ctx, req, onSegment)
	if err != nil {
		return c.jobFailed(log, u, start, req.Model, err)
	}

	text := strings.TrimSpace(res.Text)
	log.Debug("wyoming transcription", "model", res.Model, "language", res.Language,
		"audio_seconds", round2(res.Duration), "elapsed", time.Since(start).Round(time.Millisecond),
		"chars", len(text), "status", "ok")
	log.Debug("wyoming transcript", "text", text)
	if streaming && text != lastChunk {
		// Leave the chunk stream ending on the final text, so clients that
		// only keep the last chunk (wyoming's HTTP ASR server) agree with it.
		if !started {
			if c.write(typeTranscriptStart, transcriptStartData{Language: firstNonEmpty(lang, res.Language)}, nil) != nil {
				return false
			}
			started = true
		}
		if c.write(typeTranscriptChunk, transcriptChunkData{Text: text}, nil) != nil {
			return false
		}
	}
	return c.writeResult(streaming, started, text, res.Language, lang)
}

// writeResult sends transcript, wrapped in the streaming events when on.
func (c *conn) writeResult(streaming, started bool, text, language, startLang string) bool {
	if streaming && !started {
		if c.write(typeTranscriptStart, transcriptStartData{Language: firstNonEmpty(startLang, language)}, nil) != nil {
			return false
		}
	}
	if c.write(typeTranscript, transcriptData{Text: text, Language: language}, nil) != nil {
		return false
	}
	if streaming {
		if c.write(typeTranscriptStop, nil, nil) != nil {
			return false
		}
	}
	return true
}

func (c *conn) jobFailed(log *slog.Logger, u *utterance, start time.Time, model string, err error) bool {
	kind := core.KindOf(err)
	elapsed := time.Since(start).Round(time.Millisecond)
	if c.ctx.Err() != nil {
		if c.hard.Err() != nil {
			log.Debug("wyoming transcription", "model", model, "audio_seconds", round2(u.seconds),
				"elapsed", elapsed, "status", "canceled", "reason", "shutdown")
			c.sendError(core.KindUnavailable.String(), msgShutdown)
		} else {
			log.Debug("wyoming transcription", "model", model, "audio_seconds", round2(u.seconds),
				"elapsed", elapsed, "status", "canceled", "reason", "client gone")
		}
		return false
	}
	log.Debug("wyoming transcription", "model", model, "audio_seconds", round2(u.seconds),
		"elapsed", elapsed, "status", kind.String())
	log.Debug("wyoming transcription error", "err", err)
	c.sendError(kind.String(), core.ClientMessage(err))
	return false
}

// convert turns the buffered raw segments into engine PCM.
func (c *conn) convert(u *utterance) (core.PCM, error) {
	var out core.PCM
	for i, s := range u.segs {
		data := s.data
		frame := s.width * s.channels
		if rem := len(data) % frame; rem != 0 {
			// Chunks are a byte stream, so only the very end can hold a
			// partial frame; resamplers reject it.
			c.log.Debug("wyoming dropping partial frame", "utterance", u.id, "bytes", rem)
			data = data[:len(data)-rem]
		}
		if s.width == 1 {
			// Wyoming 8-bit audio is unsigned like WAV (wyoming 1.10.1 adds the
			// bias); FromRaw takes signed samples.
			for j := range data {
				data[j] ^= 0x80
			}
		}
		pcm, err := c.s.dec.FromRaw(c.ctx, data, s.rate, s.width, s.channels)
		if err != nil {
			return nil, err
		}
		c.log.Debug("wyoming audio converted", "utterance", u.id, "segment", i,
			"rate", s.rate, "width", s.width, "channels", s.channels, "in_bytes", len(data), "out_bytes", len(pcm))
		if len(u.segs) == 1 {
			return pcm, nil
		}
		out = append(out, pcm...)
	}
	return out, nil
}

func (c *conn) info() infoData {
	models := c.s.svc.Models(c.ctx)
	list := make([]asrModel, 0, len(models))
	for _, m := range models {
		desc := m.ID
		if m.Engine != "" {
			desc += " (" + m.Engine + ")"
		}
		list = append(list, asrModel{
			Name:        m.ID,
			Attribution: engineAttribution(m.Engine),
			Installed:   true,
			Description: strPtr(desc),
			// core.Model carries no version; the gateway's is the best we have.
			Version:   strPtr(c.s.cfg.Version),
			Languages: advertisedLanguages(m),
		})
	}
	return infoData{
		ASR: []asrProgram{{
			Name:        c.s.cfg.ProgramName,
			Attribution: attribution{Name: "whisper-hub", URL: "https://github.com/GRbit/whisper-hub"},
			// HA skips programs that are not installed; with no usable model
			// there is nothing it could do with this one.
			Installed:                   len(list) > 0,
			Description:                 strPtr("Whisper speech-to-text gateway"),
			Version:                     strPtr(c.s.cfg.Version),
			Models:                      list,
			SupportsTranscriptStreaming: c.s.cfg.StreamTranscripts,
			// Whisper has no endpointing: HA must detect the end of speech.
			RequiresExternalVAD:          true,
			PrefersAutoGainEnabled:       true,
			PrefersNoiseReductionEnabled: true,
		}},
		TTS:    []struct{}{},
		Handle: []struct{}{},
		Intent: []struct{}{},
		Wake:   []struct{}{},
		Mic:    []struct{}{},
		Snd:    []struct{}{},
	}
}

func engineAttribution(engine string) attribution {
	switch engine {
	case "faster_whisper":
		return attribution{Name: "SYSTRAN", URL: "https://github.com/SYSTRAN/faster-whisper"}
	case "whisperx":
		return attribution{Name: "Max Bain", URL: "https://github.com/m-bain/whisperX"}
	default:
		return attribution{Name: "OpenAI", URL: "https://github.com/openai/whisper"}
	}
}

// write sends one event; a failure closes the connection, since a partial
// frame cannot be recovered from.
func (c *conn) write(eventType string, data any, payload []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	if enabled(c.s.cfg.WriteTimeout) {
		_ = c.nc.SetWriteDeadline(time.Now().Add(c.s.cfg.WriteTimeout))
	}
	err := c.w.WriteEvent(eventType, data, payload)
	if err != nil {
		c.log.Debug("wyoming write failed", "type", eventType, "err", err)
		if c.cancel != nil {
			c.cancel()
		}
		c.close()
		return err
	}
	if eventType != typeTranscriptChunk {
		c.log.Debug("wyoming event sent", "type", eventType)
	}
	return nil
}

func (c *conn) sendError(code, text string) {
	_ = c.write(typeError, errorData{Text: text, Code: code}, nil)
}

func (c *conn) close() { c.closeOnce.Do(func() { _ = c.nc.Close() }) }

func deref(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func orAuto(lang string) string {
	if lang == "" {
		return "auto"
	}
	return lang
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func round2(f float64) float64 { return float64(int64(f*100+0.5)) / 100 }
