// Package legacy serves the HTTP API of ahmetoner/whisper-asr-webservice
// (/asr, /detect-language, /docs, /openapi.json, /assets) on top of
// core.Service. It aims at the reference's bytes where the reference is
// deterministic; what is promised is API compatibility.
//
// Transcript bodies are not formatted here: the engine worker renders them
// with the reference code (core.Request.LegacyOutput). Everything around the
// body (query and form parsing, validation errors, status codes, headers)
// reproduces public upstream, whose ground truth is captured in
// testdata/reference. whisper-hub's own extensions on top of it (the
// X-Audio-Duration-Seconds header, audio_duration_seconds in
// /detect-language, JSON details for failures and redacted 400s for audio
// that does not decode) are listed in reference_test.go.
package legacy

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Config configures the legacy frontend. The zero value is valid.
type Config struct {
	// MaxUploadBytes caps the request body of /asr and /detect-language;
	// larger requests get 413. Default 0: unlimited, as upstream.
	MaxUploadBytes int64 `yaml:"max_upload_bytes"`
	// RawSampleRate is the sample rate of encode=false uploads. Default
	// 16000.
	RawSampleRate int `yaml:"raw_sample_rate"`
}

// Server is the legacy HTTP frontend.
type Server struct {
	cfg Config
	svc core.Service
	dec core.AudioDecoder
	log *slog.Logger
}

// New returns a legacy frontend using svc for recognition and dec for
// turning uploads into PCM.
func New(cfg Config, svc core.Service, dec core.AudioDecoder) *Server {
	return &Server{cfg: cfg.Resolved(), svc: svc, dec: dec, log: slog.Default().With("protocol", core.ProtocolLegacy)}
}

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it.
func (c Config) Resolved() Config {
	if c.RawSampleRate <= 0 {
		c.RawSampleRate = core.SampleRate
	}
	return c
}

// Register adds the legacy routes to mux.
//
// It also registers the "/" catch-all, so paths nobody else claims get the
// reference's JSON 404 and its trailing-slash redirects.
func (s *Server) Register(mux *http.ServeMux) {
	mux.HandleFunc("/asr", s.handleASR)
	mux.HandleFunc("/detect-language", s.handleDetect)
	mux.HandleFunc("/docs", s.handleStatic)
	mux.HandleFunc("/docs/oauth2-redirect", s.handleStatic)
	mux.HandleFunc("/redoc", s.handleStatic)
	mux.HandleFunc("/openapi.json", s.handleStatic)
	mux.HandleFunc("/assets/", s.handleAsset)
	// Without this exact route ServeMux would answer /assets itself with a
	// relative 301; the reference sends an absolute 307.
	mux.HandleFunc("/assets", s.handleFallback)
	mux.HandleFunc("/", s.handleFallback)
}

func writeJSON(w http.ResponseWriter, status int, body []byte) {
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	w.Write(body)
}

func writeDetail(w http.ResponseWriter, status int, detail string) {
	b := appendPyJSONString([]byte(`{"detail":`), detail)
	writeJSON(w, status, append(b, '}'))
}

// writeServerError is Starlette's response to an unhandled exception.
func writeServerError(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/plain; charset=utf-8")
	h.Set("Content-Length", "21")
	w.WriteHeader(http.StatusInternalServerError)
	io.WriteString(w, "Internal Server Error")
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	if allow != "" {
		w.Header().Set("Allow", allow)
	}
	writeDetail(w, http.StatusMethodNotAllowed, "Method Not Allowed")
}

// upload is the audio_file form value FastAPI would validate: the last
// complete item with that name.
type upload struct {
	isFile   bool
	value    string // non-file value
	filename string
	size     int64
	magic    []byte // first 12 bytes, as upstream logs and reports
	pcm      core.PCM
	raw      []byte // encode=false payload
	err      error  // DecodeFile error
}

// sizeReader counts and samples the bytes of an upload as they stream by.
type sizeReader struct {
	r io.Reader
	u *upload
}

func (c *sizeReader) Read(b []byte) (int, error) {
	n, err := c.r.Read(b)
	if len(c.u.magic) < 12 {
		c.u.magic = append(c.u.magic, b[:min(n, 12-len(c.u.magic))]...)
	}
	c.u.size += int64(n)
	return n, err
}

func (u *upload) detail(reason string) string {
	return fmt.Sprintf("%s (%d bytes, magic=%s)", reason, u.size, hex.EncodeToString(u.magic))
}

// readUpload parses the request body the way Starlette's request.form()
// does and streams the audio_file part straight into the decoder, so the
// upload is never held in memory as a whole (except encode=false, where
// the raw payload is the audio). decode is false when the query already
// failed validation: the body still has to be parsed, because body errors
// win over validation errors upstream, but decoding would be wasted.
func (s *Server) readUpload(w http.ResponseWriter, r *http.Request, encode, decode bool) (*upload, error) {
	body := r.Body
	if s.cfg.MaxUploadBytes > 0 {
		body = http.MaxBytesReader(w, r.Body, s.cfg.MaxUploadBytes)
	}
	ctype, params := parseOptionsHeader(latin1Decode([]byte(r.Header.Get("Content-Type"))))
	switch ctype {
	case "multipart/form-data":
	case "application/x-www-form-urlencoded":
		pairs, err := urlencodedForm(body)
		if err != nil {
			return nil, err
		}
		var last *upload
		for _, p := range pairs {
			if p.name == "audio_file" {
				last = &upload{value: p.value}
			}
		}
		return last, nil
	default:
		// Starlette parses no other content type: the form is empty.
		return nil, nil
	}
	boundary, ok := params["boundary"]
	if !ok {
		return nil, &formError{"Missing boundary in multipart."}
	}
	charset, ok := params["charset"]
	if !ok {
		charset = "utf-8"
	}
	mr, err := newMultipartReader(body, boundary, charset)
	if err != nil {
		return nil, err
	}
	var last *upload
	for {
		p, err := mr.next()
		if err == io.EOF {
			return last, nil
		}
		if err != nil {
			return nil, err
		}
		if !p.isFile {
			var buf bytes.Buffer
			n, err := io.Copy(&buf, io.LimitReader(p.data, maxPartSize+1))
			if err == nil && n > maxPartSize {
				return nil, &formError{"Part exceeded maximum size of 1024KB."}
			}
			if err != nil {
				if errors.Is(err, errTruncated) {
					continue
				}
				return nil, err
			}
			if p.name == "audio_file" {
				last = &upload{value: decodeCharset(buf.Bytes(), charset)}
			}
			continue
		}
		if p.name != "audio_file" {
			if _, err := io.Copy(io.Discard, p.data); err != nil && !errors.Is(err, errTruncated) {
				return nil, err
			}
			continue
		}
		u := &upload{isFile: true, filename: p.filename}
		src := &sizeReader{r: p.data, u: u}
		s.log.Debug("legacy: reading upload", "path", r.URL.Path, "filename", u.filename, "encode", encode, "decode", decode)
		switch {
		case decode && encode:
			u.pcm, u.err = s.dec.DecodeFile(r.Context(), src)
		case decode:
			u.raw, _ = io.ReadAll(src)
		}
		// The decoder may stop early; upstream reports the full upload size.
		io.Copy(io.Discard, src)
		if p.data.err != nil {
			if errors.Is(p.data.err, errTruncated) {
				continue
			}
			return nil, p.data.err
		}
		last = u
	}
}

// writeBodyError answers a request whose body could not be read or parsed.
func (s *Server) writeBodyError(w http.ResponseWriter, r *http.Request, err error) {
	var fe *formError
	var mbe *http.MaxBytesError
	switch {
	case errors.As(err, &mbe):
		s.log.Info("legacy: upload too large", "path", r.URL.Path, "limit", mbe.Limit)
		writeDetail(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("Upload exceeds the maximum size of %d bytes", mbe.Limit))
	case errors.As(err, &fe):
		s.log.Debug("legacy: bad form", "path", r.URL.Path, "detail", fe.msg)
		writeDetail(w, http.StatusBadRequest, fe.msg)
	case errors.Is(err, errBodyParse):
		s.log.Debug("legacy: malformed multipart body", "path", r.URL.Path)
		writeDetail(w, http.StatusBadRequest, errBodyParse.Error())
	default:
		// Reading the body failed: the client is most likely gone.
		s.log.Debug("legacy: reading request body failed", "path", r.URL.Path, "err", err)
		writeServerError(w)
	}
}

// fileError is the body part of FastAPI's validation for audio_file.
func fileError(u *upload) *fieldError {
	loc := [2]string{"body", "audio_file"}
	switch {
	case u == nil || !u.isFile && u.value == "":
		// FastAPI treats an empty form string as missing.
		return &fieldError{typ: "missing", loc: loc, msg: msgMissing}
	case !u.isFile:
		in := u.value
		return &fieldError{typ: "value_error", loc: loc, msg: msgNotFile, input: &in, valueErrorCtx: true}
	}
	return nil
}

// audio finishes decoding the validated upload. On failure it has written
// the response and returns false.
func (s *Server) audio(w http.ResponseWriter, r *http.Request, u *upload, encode bool) (core.PCM, bool) {
	var pcm core.PCM
	if encode {
		if u.err != nil {
			if core.KindOf(u.err) == core.KindBadAudio {
				s.log.Debug("legacy: upload did not decode", "path", r.URL.Path, "bytes", u.size, "err", u.err)
				// The decoder's message is upstream's detail text verbatim.
				writeDetail(w, http.StatusBadRequest, core.ClientMessage(u.err))
			} else if r.Context().Err() == nil {
				s.log.Error("legacy: decoder failed", "path", r.URL.Path, "err", u.err)
				writeServerError(w)
			}
			return nil, false
		}
		pcm = u.pcm
	} else {
		if len(u.raw)%2 != 0 {
			writeDetail(w, http.StatusBadRequest, u.detail(fmt.Sprintf("Raw PCM payload has an odd length (%d bytes)", len(u.raw))))
			return nil, false
		}
		if len(u.raw) > 0 {
			var err error
			pcm, err = s.dec.FromRaw(r.Context(), u.raw, s.cfg.RawSampleRate, 2, 1)
			u.raw = nil
			if err != nil {
				s.log.Error("legacy: raw PCM conversion failed", "path", r.URL.Path, "err", err)
				writeServerError(w)
				return nil, false
			}
		}
	}
	if len(pcm)%2 != 0 {
		writeDetail(w, http.StatusBadRequest, u.detail(fmt.Sprintf("Raw PCM payload has an odd length (%d bytes)", len(pcm))))
		return nil, false
	}
	if len(pcm) == 0 {
		writeDetail(w, http.StatusBadRequest, u.detail("Audio decoded to 0 samples"))
		return nil, false
	}
	s.log.Debug("legacy: decoded upload", "path", r.URL.Path, "bytes", u.size, "seconds", pcm.Seconds())
	return pcm, true
}

const durationHeader = "X-Audio-Duration-Seconds"

func durationValue(pcm core.PCM) string {
	return strconv.FormatFloat(float64(len(pcm)/2)/core.SampleRate, 'f', 3, 64)
}

// writeServiceError maps a core.Service failure. Upstream answers every
// engine exception with 500 and a fixed detail, so bad_request (for
// example an unknown language the worker rejected) maps there too; the
// kinds upstream cannot produce get their natural status.
func (s *Server) writeServiceError(w http.ResponseWriter, r *http.Request, err error, duration, failed string) {
	w.Header().Set(durationHeader, duration)
	kind := core.KindOf(err)
	s.log.Debug("legacy: job failed", "path", r.URL.Path, "kind", kind.String(), "err", err)
	switch kind {
	case core.KindCanceled:
		if r.Context().Err() != nil {
			return
		}
		writeDetail(w, http.StatusInternalServerError, failed)
	case core.KindUnauthenticated:
		writeDetail(w, http.StatusUnauthorized, core.ClientMessage(err))
	case core.KindForbidden:
		writeDetail(w, http.StatusForbidden, core.ClientMessage(err))
	case core.KindModelNotFound:
		writeDetail(w, http.StatusNotFound, core.ClientMessage(err))
	case core.KindUnavailable:
		writeDetail(w, http.StatusServiceUnavailable, core.ClientMessage(err))
	default:
		writeDetail(w, http.StatusInternalServerError, failed)
	}
}

// model returns the model a request for name would run on: the one with
// that id or alias, else the default (the hub's fallback for unknown names).
func (s *Server) model(ctx context.Context, name string) core.Model {
	models := s.svc.Models(ctx)
	if name != "" {
		for _, m := range models {
			if m.ID == name {
				return m
			}
			for _, a := range m.Aliases {
				if a == name {
					return m
				}
			}
		}
	}
	for _, m := range models {
		if m.Default {
			return m
		}
	}
	if len(models) > 0 {
		return models[0]
	}
	return core.Model{}
}

func (s *Server) handleASR(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	v := validator{q: parseQuery(r.URL.RawQuery)}
	encode := v.boolean("encode", true)
	task, _ := v.str("task", "transcribe")
	language, _ := v.str("language", "")
	prompt, _ := v.str("initial_prompt", "")
	vad := v.boolean("vad_filter", false)
	words := v.boolean("word_timestamps", false)
	diarize := v.boolean("diarize", false)
	minSpeakers := v.integer("min_speakers")
	maxSpeakers := v.integer("max_speakers")
	output, _ := v.str("output", "txt")
	modelName, _ := v.str("model", "")
	s.log.Debug("legacy: asr request", "query", r.URL.RawQuery, "content_type", r.Header.Get("Content-Type"))

	u, err := s.readUpload(w, r, encode, len(v.errs) == 0)
	if err != nil {
		s.writeBodyError(w, r, err)
		return
	}
	if fe := fileError(u); fe != nil {
		v.errs = append(v.errs, *fe)
	}
	if len(v.errs) > 0 {
		s.log.Debug("legacy: asr validation failed", "errors", len(v.errs))
		writeJSON(w, http.StatusUnprocessableEntity, renderValidationErrors(v.errs))
		return
	}
	// Starlette encodes header values as latin-1 and fails with a bare 500
	// when it cannot; the reference only finds out after transcribing.
	disposition, ok := latin1Encode(`attachment; filename="` + pyQuote(u.filename) + "." + output + `"`)
	if !ok {
		s.log.Info("legacy: output parameter cannot be sent in a header", "output", output)
		writeServerError(w)
		return
	}
	pcm, ok := s.audio(w, r, u, encode)
	if !ok {
		return
	}
	duration := durationValue(pcm)

	m := s.model(r.Context(), modelName)
	req := &core.Request{
		Audio:    pcm,
		Model:    modelName,
		Task:     core.Task(task),
		Language: language,
		Prompt:   prompt,
		// The hub drops options the engine does not support, as each
		// upstream engine silently ignores them.
		VADFilter:      vad,
		WordTimestamps: words,
		Diarize:        diarize,
		MinSpeakers:    minSpeakers,
		MaxSpeakers:    maxSpeakers,
		// Upstream renders unknown output values as txt but still names
		// the file after them.
		LegacyOutput: legacyOutput(output),
		Priority:     core.PriorityNormal,
		Protocol:     core.ProtocolLegacy,
		Endpoint:     "/asr",
	}
	s.log.Debug("legacy: transcribing", "model", modelName, "engine", m.Engine, "task", task,
		"language", language, "output", output, "seconds", pcm.Seconds())
	res, err := s.svc.Transcribe(r.Context(), req, nil)
	if err != nil {
		s.writeServiceError(w, r, err, duration, "Transcription failed")
		return
	}
	engine := res.Engine
	if engine == "" {
		engine = m.Engine
	}
	s.log.Debug("legacy: asr done", "model", res.Model, "engine", engine, "seconds", duration, "output", output)
	h := w.Header()
	h.Set("Asr-Engine", engine)
	h.Set(durationHeader, duration)
	h["Content-Disposition"] = []string{string(disposition)}
	h.Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	// Upstream streams the body; flushing first gives the same chunked
	// framing instead of a Content-Length.
	http.NewResponseController(w).Flush()
	w.Write(res.Rendered)
}

func legacyOutput(output string) string {
	switch output {
	case "txt", "vtt", "srt", "tsv", "json":
		return output
	}
	return "txt"
}

func (s *Server) handleDetect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, "POST")
		return
	}
	v := validator{q: parseQuery(r.URL.RawQuery)}
	encode := v.boolean("encode", true)
	modelName, _ := v.str("model", "")
	s.log.Debug("legacy: detect-language request", "query", r.URL.RawQuery, "content_type", r.Header.Get("Content-Type"))

	u, err := s.readUpload(w, r, encode, len(v.errs) == 0)
	if err != nil {
		s.writeBodyError(w, r, err)
		return
	}
	if fe := fileError(u); fe != nil {
		v.errs = append(v.errs, *fe)
	}
	if len(v.errs) > 0 {
		writeJSON(w, http.StatusUnprocessableEntity, renderValidationErrors(v.errs))
		return
	}
	pcm, ok := s.audio(w, r, u, encode)
	if !ok {
		return
	}
	duration := durationValue(pcm)
	det, err := s.svc.DetectLanguage(r.Context(), &core.DetectRequest{
		Audio:    pcm,
		Model:    modelName,
		Priority: core.PriorityNormal,
		Protocol: core.ProtocolLegacy,
		Endpoint: "/detect-language",
	})
	if err != nil {
		s.writeServiceError(w, r, err, duration, "Language detection failed")
		return
	}
	name := core.LanguageName(det.Language)
	confidence, okConf := pyFloatRepr(det.Confidence)
	seconds, _ := strconv.ParseFloat(duration, 64)
	secondsRepr, _ := pyFloatRepr(seconds)
	if name == "" || !okConf {
		// Upstream fails the same way: a KeyError on the language table or
		// a JSON encoder that refuses NaN.
		s.log.Error("legacy: unusable detection result", "language", det.Language, "confidence", det.Confidence)
		writeServerError(w)
		return
	}
	s.log.Debug("legacy: detect-language done", "model", det.Model, "language", det.Language, "seconds", duration)
	b := appendPyJSONString([]byte(`{"detected_language":`), name)
	b = appendPyJSONString(append(b, `,"language_code":`...), det.Language)
	b = append(append(b, `,"confidence":`...), confidence...)
	b = append(append(b, `,"audio_duration_seconds":`...), secondsRepr...)
	b = append(b, '}')
	w.Header().Set(durationHeader, duration)
	writeJSON(w, http.StatusOK, b)
}

// knownRoute reports whether the reference app has a route for path, with
// any method; it drives Starlette's redirect_slashes.
func knownRoute(path string) bool {
	switch path {
	case "/", "/asr", "/detect-language", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json":
		return true
	}
	return strings.HasPrefix(path, "/assets/")
}

func (s *Server) handleFallback(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	if path == "/" {
		if r.Method != http.MethodGet {
			methodNotAllowed(w, "GET")
			return
		}
		redirect(w, "/docs")
		return
	}
	alt := path + "/"
	if strings.HasSuffix(path, "/") {
		alt = strings.TrimSuffix(path, "/")
	}
	if knownRoute(alt) {
		scheme := "http"
		if r.TLS != nil {
			scheme = "https"
		}
		target := scheme + "://" + r.Host + alt
		if r.URL.RawQuery != "" {
			target += "?" + r.URL.RawQuery
		}
		s.log.Debug("legacy: trailing slash redirect", "from", path, "to", alt)
		redirect(w, target)
		return
	}
	s.log.Debug("legacy: unknown path", "method", r.Method, "path", path)
	writeDetail(w, http.StatusNotFound, "Not Found")
}

func redirect(w http.ResponseWriter, location string) {
	w.Header().Set("Location", location)
	w.Header().Set("Content-Length", "0")
	w.WriteHeader(http.StatusTemporaryRedirect)
}
