package wyoming

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

type call struct {
	req *core.Request
	id  *core.Identity
}

// fakeService records requests and answers from segments, or from fn when
// set. Transcribe sends each segment through onSegment.
type fakeService struct {
	mu       sync.Mutex
	models   []core.Model
	calls    []call
	modelIDs []*core.Identity
	segments []string
	fn       func(ctx context.Context, req *core.Request) (*core.Result, error)
}

func newFakeService() *fakeService {
	return &fakeService{
		models: []core.Model{
			{ID: "tiny", Engine: "faster_whisper", Languages: []string{"en", "nl", "pt", "yue", "zh"}, Default: true},
			{ID: "small", Engine: "whisperx", Languages: []string{"en", "de"}},
		},
		segments: []string{" Hello", " world."},
	}
}

func (f *fakeService) Transcribe(ctx context.Context, req *core.Request, onSegment func(core.Segment) error) (*core.Result, error) {
	f.mu.Lock()
	f.calls = append(f.calls, call{req: req, id: core.IdentityFrom(ctx)})
	segs, fn := f.segments, f.fn
	f.mu.Unlock()
	if fn != nil {
		return fn(ctx, req)
	}
	var text strings.Builder
	for i, s := range segs {
		text.WriteString(s)
		if onSegment != nil {
			if err := onSegment(core.Segment{ID: i, Text: s}); err != nil {
				return nil, err
			}
		}
	}
	lang := req.Language
	if lang == "" {
		lang = "en"
	}
	return &core.Result{Model: "tiny", Engine: "faster_whisper", Language: lang, Duration: req.Audio.Seconds(), Text: text.String()}, nil
}

func (f *fakeService) DetectLanguage(context.Context, *core.DetectRequest) (*core.Detection, error) {
	return nil, errors.New("not used")
}

func (f *fakeService) Models(ctx context.Context) []core.Model {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.modelIDs = append(f.modelIDs, core.IdentityFrom(ctx))
	return append([]core.Model(nil), f.models...)
}

func (f *fakeService) Calls() []call {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]call(nil), f.calls...)
}

type rawCall struct {
	rate, width, channels int
	data                  []byte
}

// fakeDecoder returns 16 kHz input unchanged and otherwise a buffer of the
// right length, so tests can check durations without a real resampler.
type fakeDecoder struct {
	mu    sync.Mutex
	calls []rawCall
	err   error
}

func (d *fakeDecoder) DecodeFile(context.Context, io.Reader) (core.PCM, error) {
	return nil, errors.New("not used")
}

func (d *fakeDecoder) FromRaw(_ context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, rawCall{rate, width, channels, append([]byte(nil), data...)})
	if d.err != nil {
		return nil, d.err
	}
	if rate == core.SampleRate && width == 2 && channels == 1 {
		return core.PCM(append([]byte(nil), data...)), nil
	}
	frames := len(data) / (width * channels)
	return make(core.PCM, frames*core.SampleRate/rate*2), nil
}

func (d *fakeDecoder) Calls() []rawCall {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]rawCall(nil), d.calls...)
}

type harness struct {
	t      *testing.T
	svc    *fakeService
	dec    *fakeDecoder
	addr   string
	cancel context.CancelFunc
	done   chan error
}

func testLogger() *slog.Logger {
	if os.Getenv("WYOMING_TEST_LOG") != "" {
		return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))
	}
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func startServer(t *testing.T, cfg Config, svc *fakeService, identify IdentifyFunc) *harness {
	t.Helper()
	dec := &fakeDecoder{}
	srv, err := New(cfg, svc, dec, identify, testLogger())
	if err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	h := &harness{t: t, svc: svc, dec: dec, addr: ln.Addr().String(), cancel: cancel, done: make(chan error, 1)}
	go func() { h.done <- srv.Serve(ctx, ln) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-h.done:
		case <-time.After(5 * time.Second):
			t.Error("Serve did not return after cancel")
		}
	})
	return h
}

type client struct {
	t  *testing.T
	nc net.Conn
	r  *frameReader
	w  *frameWriter
}

func (h *harness) dial() *client {
	h.t.Helper()
	nc, err := net.Dial("tcp", h.addr)
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { _ = nc.Close() })
	return &client{t: h.t, nc: nc, r: newFrameReader(nc, limits{}), w: newFrameWriter(nc)}
}

func (c *client) send(typ string, data any, payload []byte) {
	c.t.Helper()
	if err := c.w.WriteEvent(typ, data, payload); err != nil {
		c.t.Fatalf("send %s: %v", typ, err)
	}
}

func (c *client) recv() *event {
	c.t.Helper()
	_ = c.nc.SetReadDeadline(time.Now().Add(5 * time.Second))
	ev, err := c.r.ReadEvent()
	if err != nil {
		c.t.Fatalf("recv: %v", err)
	}
	return ev
}

// expectClosed asserts the server closes the connection without sending
// anything else.
func (c *client) expectClosed() {
	c.t.Helper()
	_ = c.nc.SetReadDeadline(time.Now().Add(5 * time.Second))
	ev, err := c.r.ReadEvent()
	if err == nil {
		c.t.Fatalf("want close, got event %s %v", ev.Type, ev.Data)
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		c.t.Fatal("connection still open")
	}
}

func (c *client) utterance(rate, width, channels int, chunks ...[]byte) {
	c.t.Helper()
	f := map[string]any{"rate": rate, "width": width, "channels": channels, "timestamp": nil}
	c.send(typeAudioStart, f, nil)
	for _, ch := range chunks {
		c.send(typeAudioChunk, f, ch)
	}
	c.send(typeAudioStop, map[string]any{"timestamp": nil}, nil)
}

func decode[T any](t *testing.T, ev *event) T {
	t.Helper()
	var v T
	if err := ev.decodeData(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

func pcm(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
