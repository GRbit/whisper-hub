// Package workerclient talks to one Python engine worker over the internal
// Worker API described in docs/worker-api.md.
package workerclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"net/textproto"
	"net/url"
	"strings"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Config tunes the HTTP client. Zero values select the documented defaults.
type Config struct {
	// ConnectTimeout bounds TCP connection setup. Default 5s.
	ConnectTimeout time.Duration `yaml:"connect_timeout"`
	// FirstByteTimeout bounds the time from sending a job until the first
	// NDJSON line (or the HTTP error) arrives. It must cover lazy model
	// loading on the worker. Default 10m.
	FirstByteTimeout time.Duration `yaml:"first_byte_timeout"`
	// RequestTimeout is the overall deadline of the small calls (/healthz,
	// /v1/info) and of /v1/detect-language. Default 30s.
	RequestTimeout time.Duration `yaml:"request_timeout"`
	// MaxLineBytes caps one NDJSON line; a "done" line carries the full
	// text and rendered output, so it is generous. Default 64 MiB.
	MaxLineBytes int `yaml:"max_line_bytes"`
}

// Defaults for Config fields.
const (
	DefaultConnectTimeout   = 5 * time.Second
	DefaultFirstByteTimeout = 10 * time.Minute
	DefaultRequestTimeout   = 30 * time.Second
	DefaultMaxLineBytes     = 64 << 20
)

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it.
func (c Config) Resolved() Config {
	if c.ConnectTimeout <= 0 {
		c.ConnectTimeout = DefaultConnectTimeout
	}
	if c.FirstByteTimeout <= 0 {
		c.FirstByteTimeout = DefaultFirstByteTimeout
	}
	if c.RequestTimeout <= 0 {
		c.RequestTimeout = DefaultRequestTimeout
	}
	if c.MaxLineBytes <= 0 {
		c.MaxLineBytes = DefaultMaxLineBytes
	}
	return c
}

// Client is a Worker API client for one worker. It is safe for concurrent use.
type Client struct {
	base string
	cfg  Config
	http *http.Client
	log  *slog.Logger
}

// New returns a client for the worker at baseURL (for example
// "http://worker:9100"). A nil log uses slog.Default().
func New(baseURL string, cfg Config, log *slog.Logger) (*Client, error) {
	u, err := url.Parse(baseURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("workerclient: invalid worker url %q", baseURL)
	}
	if log == nil {
		log = slog.Default()
	}
	cfg = cfg.Resolved()
	dialer := &net.Dialer{Timeout: cfg.ConnectTimeout, KeepAlive: 30 * time.Second}
	tr := &http.Transport{
		Proxy:               nil, // internal traffic must never go through HTTP_PROXY
		DialContext:         dialer.DialContext,
		MaxIdleConnsPerHost: 8,
		IdleConnTimeout:     90 * time.Second,
		// No ResponseHeaderTimeout and no Client.Timeout: a long file can
		// legitimately stream for an hour. FirstByteTimeout is enforced per call.
	}
	return &Client{
		base: strings.TrimRight(u.String(), "/"),
		cfg:  cfg,
		http: &http.Client{Transport: tr},
		log:  log.With("worker", baseURL),
	}, nil
}

// URL returns the worker base URL.
func (c *Client) URL() string { return c.base }

// Close releases idle connections.
func (c *Client) Close() { c.http.CloseIdleConnections() }

// Health checks GET /healthz.
func (c *Client) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	resp, err := c.do(ctx, http.MethodGet, "/healthz", nil, "", 0)
	if err != nil {
		return err
	}
	defer drain(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return httpError(resp)
	}
	return nil
}

// Info fetches GET /v1/info.
func (c *Client) Info(ctx context.Context) (*Info, error) {
	ctx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	resp, err := c.do(ctx, http.MethodGet, "/v1/info", nil, "", 0)
	if err != nil {
		return nil, err
	}
	defer drain(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, httpError(resp)
	}
	var info Info
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&info); err != nil {
		return nil, core.WrapError(core.KindInternal, err, "engine worker sent an invalid info response")
	}
	return &info, nil
}

// DetectLanguage calls POST /v1/detect-language. An empty model means the
// worker default.
func (c *Client) DetectLanguage(ctx context.Context, model string, audio core.PCM) (*core.Detection, error) {
	opts := struct {
		Model *string `json:"model"`
	}{}
	if model != "" {
		opts.Model = &model
	}
	ctx, cancel := context.WithTimeout(ctx, max(c.cfg.RequestTimeout, c.cfg.FirstByteTimeout))
	defer cancel()
	started := time.Now()
	resp, err := c.postJob(ctx, "/v1/detect-language", opts, audio)
	if err != nil {
		return nil, err
	}
	defer drain(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, httpError(resp)
	}
	var b detectBody
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&b); err != nil {
		return nil, mapStreamErr(ctx, err, "engine worker sent an invalid detect-language response")
	}
	c.log.Debug("detect-language done", "language", b.Language, "confidence", b.Confidence,
		"model", b.Model, "elapsed", time.Since(started))
	return &core.Detection{Language: b.Language, Confidence: b.Confidence, Engine: b.Engine, Model: b.Model}, nil
}

// Transcribe calls POST /v1/transcribe and consumes the NDJSON stream.
// onSegment, when not nil, runs for every segment as it arrives; an error
// from it aborts the job (the HTTP request is canceled, which tells the
// worker to stop) and is returned as KindCanceled unless it already is a
// *core.Error. onPause, when not nil, runs with true when the worker
// reports the job paused (see Pause) and with false when it runs again.
func (c *Client) Transcribe(ctx context.Context, opts Options, audio core.PCM, onSegment func(core.Segment) error,
	onPause func(paused bool)) (*core.Result, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	errFirstByte := errors.New("first byte timeout")
	timer := time.AfterFunc(c.cfg.FirstByteTimeout, func() { cancel(errFirstByte) })
	defer timer.Stop()

	started := time.Now()
	resp, err := c.postJob(ctx, "/v1/transcribe", opts, audio)
	if err != nil {
		if errors.Is(context.Cause(ctx), errFirstByte) {
			return nil, core.WrapError(core.KindUnavailable, err, "engine worker did not respond in time")
		}
		return nil, err
	}
	defer drain(resp.Body)
	if resp.StatusCode != http.StatusOK {
		timer.Stop()
		return nil, httpError(resp)
	}

	res := &core.Result{}
	br := bufio.NewReaderSize(resp.Body, 64<<10)
	gotInfo := false
	for {
		line, err := readLine(br, c.cfg.MaxLineBytes)
		if err != nil {
			if errors.Is(context.Cause(ctx), errFirstByte) {
				return nil, core.WrapError(core.KindUnavailable, err, "engine worker did not respond in time")
			}
			if errors.Is(err, io.EOF) {
				return nil, core.WrapError(core.KindInternal, io.ErrUnexpectedEOF, "engine stream ended unexpectedly")
			}
			if errors.Is(err, errLineTooLong) {
				return nil, core.WrapError(core.KindInternal, err, "engine worker sent an oversized stream line")
			}
			return nil, mapStreamErr(ctx, err, "engine stream failed")
		}
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		timer.Stop()
		var ev event
		if err := unmarshalLenient(line, &ev); err != nil {
			return nil, core.WrapError(core.KindInternal, err, "engine worker sent an invalid stream line")
		}
		switch ev.Type {
		case "info":
			gotInfo = true
			res.Model, res.Engine = ev.Model, ev.Engine
			res.Language, res.LanguageProbability, res.Duration = ev.Language, ev.LanguageProbability, ev.Duration
			c.log.Debug("transcribe info", "model", ev.Model, "engine", ev.Engine, "language", ev.Language,
				"duration", ev.Duration, "first_byte", time.Since(started))
		case "segment":
			if ev.Segment == nil {
				return nil, core.Errorf(core.KindInternal, "engine worker sent an empty segment")
			}
			seg := *ev.Segment
			res.Segments = append(res.Segments, seg)
			c.log.Debug("transcribe segment", "id", seg.ID, "start", seg.Start, "end", seg.End)
			if onSegment != nil {
				if err := onSegment(seg); err != nil {
					cancel(err)
					var ce *core.Error
					if errors.As(err, &ce) {
						return nil, err
					}
					return nil, core.WrapError(core.KindCanceled, err, "segment consumer failed")
				}
			}
		case "done":
			if !gotInfo {
				c.log.Debug("transcribe stream had no info line")
			}
			res.Text = ev.Text
			if ev.Language != "" {
				res.Language = ev.Language
			}
			if ev.Rendered != nil {
				res.Rendered = []byte(*ev.Rendered)
			}
			c.log.Debug("transcribe done", "segments", len(res.Segments), "elapsed", time.Since(started))
			return res, nil
		case "error":
			c.log.Debug("transcribe stream error", "code", ev.Code, "message", ev.Message)
			return nil, codeError(ev.Code, ev.Message)
		case "paused", "resumed":
			c.log.Debug("transcribe "+ev.Type, "job", opts.JobID, "elapsed", time.Since(started))
			if onPause != nil {
				onPause(ev.Type == "paused")
			}
		default:
			c.log.Debug("ignoring unknown stream event", "type", ev.Type)
		}
	}
}

// ErrNotPausable is the cause of a Pause the worker refused at once: its
// engine cannot pause, or the model to run next would not fit beside the
// paused one.
var ErrNotPausable = errors.New("worker cannot pause this job")

// Pause asks the worker to pause the job named id (Options.JobID) at its
// next pause point, so that a job for model can run meanwhile, and keeps
// it paused until ctx is done. The job may not have reached the worker
// yet. Whether and when the pause takes effect is reported in that job's
// stream, not here. Pause returns nil when ctx ends the pause or the
// worker ends it because the job ended, and an error when the worker
// refused (wrapping ErrNotPausable) or could not be reached.
func (c *Client) Pause(ctx context.Context, id, model string) error {
	body, err := json.Marshal(map[string]string{"job_id": id, "model": model})
	if err != nil {
		return core.WrapError(core.KindInternal, err, "cannot encode pause request")
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/pause", func() io.Reader { return bytes.NewReader(body) },
		"application/json", int64(len(body)))
	if err != nil {
		if ctx.Err() != nil {
			return nil
		}
		return err
	}
	// Not drain: the body stays open for as long as the pause lasts.
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		err := httpError(resp)
		if resp.StatusCode == http.StatusConflict {
			return fmt.Errorf("%w: %w", ErrNotPausable, err)
		}
		return err
	}
	c.log.Debug("pause held", "job", id, "for_model", model)
	_, _ = io.Copy(io.Discard, resp.Body)
	c.log.Debug("pause released", "job", id, "by_gateway", ctx.Err() != nil)
	return nil
}

func (c *Client) postJob(ctx context.Context, path string, opts any, audio core.PCM) (*http.Response, error) {
	optJSON, err := json.Marshal(opts)
	if err != nil {
		return nil, core.WrapError(core.KindInternal, err, "cannot encode worker options")
	}
	head, tail, ctype := multipartFrame(optJSON)
	size := int64(len(head) + len(audio) + len(tail))
	c.log.Debug("worker call", "path", path, "options", string(optJSON), "audio_bytes", len(audio))
	return c.do(ctx, http.MethodPost, path, func() io.Reader {
		// Readers over the caller's slices: the PCM is never copied whole.
		return io.MultiReader(bytes.NewReader(head), bytes.NewReader(audio), bytes.NewReader(tail))
	}, ctype, size)
}

// multipartFrame renders everything of the two-part body except the audio
// bytes, so the body can be streamed with a known Content-Length.
func multipartFrame(optJSON []byte) (head, tail []byte, contentType string) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	h := textproto.MIMEHeader{}
	h.Set("Content-Disposition", `form-data; name="options"`)
	h.Set("Content-Type", "application/json")
	pw, _ := w.CreatePart(h) // writes to a bytes.Buffer cannot fail
	_, _ = pw.Write(optJSON)
	h = textproto.MIMEHeader{}
	h.Set("Content-Disposition", `form-data; name="audio"; filename="audio.pcm"`)
	h.Set("Content-Type", "application/octet-stream")
	_, _ = w.CreatePart(h)
	n := buf.Len()
	_ = w.Close()
	all := buf.Bytes()
	return all[:n:n], all[n:], w.FormDataContentType()
}

func (c *Client) do(ctx context.Context, method, path string, body func() io.Reader, ctype string, size int64) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		rd = body()
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, rd)
	if err != nil {
		return nil, core.WrapError(core.KindInternal, err, "cannot build worker request")
	}
	if body != nil {
		req.ContentLength = size
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(body()), nil }
		req.Header.Set("Content-Type", ctype)
	}
	started := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		c.log.Debug("worker call failed", "method", method, "path", path, "err", err)
		return nil, mapStreamErr(ctx, err, "engine worker unavailable")
	}
	c.log.Debug("worker response", "method", method, "path", path, "status", resp.StatusCode,
		"elapsed", time.Since(started))
	return resp, nil
}

// mapStreamErr classifies transport errors: a canceled caller is
// KindCanceled, anything else means the worker is unreachable.
func mapStreamErr(ctx context.Context, err error, msg string) error {
	if ctx.Err() != nil {
		if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, context.Canceled) && !errors.Is(cause, context.DeadlineExceeded) {
			return core.WrapError(core.KindUnavailable, err, "engine worker did not respond in time")
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return core.WrapError(core.KindUnavailable, err, "engine worker did not respond in time")
		}
		return core.WrapError(core.KindCanceled, err, "request canceled")
	}
	return core.WrapError(core.KindUnavailable, err, "%s", msg)
}

func httpError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var b errorBody
	if json.Unmarshal(raw, &b) == nil && b.Error.Code != "" {
		return codeError(b.Error.Code, b.Error.Message)
	}
	code := "internal"
	switch resp.StatusCode {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		code = "bad_request"
	case http.StatusNotFound:
		code = "model_not_found"
	case http.StatusServiceUnavailable, http.StatusTooManyRequests:
		code = "unavailable"
	}
	return codeError(code, fmt.Sprintf("http %d: %.200s", resp.StatusCode, raw))
}

// codeError maps a Worker API error code to a core.Error. The worker text
// is kept as the cause only, except for bad_request where it tells the
// client what to fix.
func codeError(code, message string) error {
	cause := fmt.Errorf("worker error %s: %s", code, message)
	switch code {
	case "bad_request":
		msg := "engine rejected the request"
		if message != "" {
			msg += ": " + truncate(message, 200)
		}
		return core.WrapError(core.KindBadRequest, cause, "%s", msg)
	case "bad_audio":
		return core.WrapError(core.KindBadAudio, cause, "engine could not process the audio")
	case "model_not_found":
		return core.WrapError(core.KindModelNotFound, cause, "model not found on engine worker")
	case "unavailable":
		return core.WrapError(core.KindUnavailable, cause, "engine worker unavailable")
	case "canceled":
		return core.WrapError(core.KindCanceled, cause, "request canceled")
	default:
		return core.WrapError(core.KindInternal, cause, "engine worker failed")
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

var errLineTooLong = errors.New("stream line too long")

// unmarshalLenient is json.Unmarshal that also accepts the bare NaN,
// Infinity and -Infinity tokens Python's json.dumps emits (for example an
// avg_logprob of -inf), reading them as 0 instead of failing a long job.
// The rewrite only runs after a normal parse failed, so valid lines cost
// nothing extra.
func unmarshalLenient(line []byte, v any) error {
	err := json.Unmarshal(line, v)
	if err == nil {
		return nil
	}
	var se *json.SyntaxError
	if !errors.As(err, &se) {
		return err
	}
	fixed, changed := replaceNonFinite(line)
	if !changed {
		return err
	}
	return json.Unmarshal(fixed, v)
}

func replaceNonFinite(b []byte) ([]byte, bool) {
	var out []byte
	inStr, esc, last := false, false, 0
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
			continue
		}
		n := 0
		for _, tok := range []string{"NaN", "Infinity", "-Infinity"} {
			if bytes.HasPrefix(b[i:], []byte(tok)) {
				n = len(tok)
				break
			}
		}
		if n == 0 {
			continue
		}
		out = append(out, b[last:i]...)
		out = append(out, '0')
		i += n - 1
		last = i + 1
	}
	if out == nil {
		return b, false
	}
	return append(out, b[last:]...), true
}

// readLine returns one line without its terminator; the final line may lack
// a newline. Lines longer than limit are an error.
func readLine(br *bufio.Reader, limit int) ([]byte, error) {
	var acc []byte
	for {
		chunk, err := br.ReadSlice('\n')
		if err == nil || (errors.Is(err, io.EOF) && len(chunk) > 0) {
			if acc == nil {
				return bytes.TrimRight(chunk, "\r\n"), nil
			}
			acc = append(acc, chunk...)
			return bytes.TrimRight(acc, "\r\n"), nil
		}
		if !errors.Is(err, bufio.ErrBufferFull) {
			if len(acc) > 0 && errors.Is(err, io.EOF) {
				return acc, nil
			}
			return nil, err
		}
		acc = append(acc, chunk...)
		if len(acc) > limit {
			return nil, fmt.Errorf("%w: limit %d bytes", errLineTooLong, limit)
		}
	}
}

func drain(rc io.ReadCloser) {
	// Bounded drain so the connection can be reused without reading an
	// abandoned hour-long stream to the end.
	_, _ = io.CopyN(io.Discard, rc, 64<<10)
	_ = rc.Close()
}
