package legacy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// These tests cover what the reference captures cannot: how requests are
// mapped onto core.Service, error kinds upstream never produced, the
// upload limit and streaming. All go through the HTTP interface.

func newTestServer(t *testing.T, cfg Config, svc *fakeService, dec core.AudioDecoder) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	New(cfg, svc, dec).Register(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, url string, body []byte) (*http.Response, string) {
	t.Helper()
	resp, err := http.Post(url, mpType, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp, string(b)
}

func TestASRRequestMapping(t *testing.T) {
	svc := newFakeService()
	svc.rendered = []byte("hello\n")
	srv := newTestServer(t, Config{}, svc, fakeDecoder{})
	q := "?task=translate&language=de&initial_prompt=Hi&vad_filter=1&word_timestamps=yes" +
		"&diarize=true&min_speakers=2&max_speakers=3&output=foo&model=whisper-1"
	resp, body := post(t, srv.URL+"/asr"+q, wavUpload(t))
	if resp.StatusCode != 200 || body != "hello\n" {
		t.Fatalf("got %d %q", resp.StatusCode, body)
	}
	if len(svc.requests) != 1 {
		t.Fatalf("%d requests", len(svc.requests))
	}
	r := svc.requests[0]
	if r.Task != core.TaskTranslate || r.Language != "de" || r.Prompt != "Hi" || !r.VADFilter || !r.WordTimestamps ||
		!r.Diarize || r.MinSpeakers == nil || *r.MinSpeakers != 2 || r.MaxSpeakers == nil || *r.MaxSpeakers != 3 {
		t.Errorf("options not mapped: %+v", r)
	}
	// Unknown outputs render as txt upstream; the filename keeps the value.
	if r.LegacyOutput != "txt" || resp.Header.Get("Content-Disposition") != `attachment; filename="english-front-center.wav.foo"` {
		t.Errorf("output: %q, %q", r.LegacyOutput, resp.Header.Get("Content-Disposition"))
	}
	if r.Model != "whisper-1" || r.Protocol != core.ProtocolLegacy || r.Endpoint != "/asr" {
		t.Errorf("routing fields: %+v", r)
	}
	if want := 22848 * 2; len(r.Audio) != want {
		t.Errorf("audio %d bytes, want %d", len(r.Audio), want)
	}
	if r.Temperature != nil {
		t.Errorf("temperature set")
	}
}

func TestASRDefaults(t *testing.T) {
	svc := newFakeService()
	srv := newTestServer(t, Config{}, svc, fakeDecoder{})
	post(t, srv.URL+"/asr", wavUpload(t))
	r := svc.requests[0]
	if r.Task != core.TaskTranscribe || r.Language != "" || r.Model != "" || r.LegacyOutput != "txt" ||
		r.VADFilter || r.WordTimestamps || r.Diarize || r.MinSpeakers != nil || r.MaxSpeakers != nil {
		t.Errorf("defaults: %+v", r)
	}
}

// rawDecoder records how FromRaw was called.
type rawDecoder struct {
	fakeDecoder
	rate, width, channels, n int
}

func (d *rawDecoder) FromRaw(_ context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	d.rate, d.width, d.channels, d.n = rate, width, channels, len(data)
	return core.PCM(data[:len(data)/2]), nil
}

func TestEncodeFalseUsesFromRaw(t *testing.T) {
	dec := &rawDecoder{}
	svc := newFakeService()
	srv := newTestServer(t, Config{RawSampleRate: 8000}, svc, dec)
	pcm := make([]byte, 16000)
	resp, _ := post(t, srv.URL+"/asr?encode=false", multipartBody(filePart("audio_file", "a.pcm", pcm)))
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if dec.rate != 8000 || dec.width != 2 || dec.channels != 1 || dec.n != 16000 {
		t.Errorf("FromRaw(%d, %d, %d) with %d bytes", dec.rate, dec.width, dec.channels, dec.n)
	}
	// The duration describes the converted PCM: 8000 bytes at 16 kHz.
	if got := resp.Header.Get(durationHeader); got != "0.250" {
		t.Errorf("duration %q", got)
	}
}

func TestServiceErrors(t *testing.T) {
	cases := []struct {
		kind   core.ErrorKind
		status int
		asr    string
		detect string
	}{
		{core.KindInternal, 500, `{"detail":"Transcription failed"}`, `{"detail":"Language detection failed"}`},
		{core.KindBadRequest, 500, `{"detail":"Transcription failed"}`, `{"detail":"Language detection failed"}`},
		{core.KindForbidden, 403, `{"detail":"engine failed"}`, `{"detail":"engine failed"}`},
		{core.KindModelNotFound, 404, `{"detail":"engine failed"}`, `{"detail":"engine failed"}`},
		{core.KindUnavailable, 503, `{"detail":"engine failed"}`, `{"detail":"engine failed"}`},
		{core.KindUnauthenticated, 401, `{"detail":"engine failed"}`, `{"detail":"engine failed"}`},
	}
	for _, c := range cases {
		t.Run(c.kind.String(), func(t *testing.T) {
			svc := newFakeService()
			svc.fail = kind(c.kind)
			srv := newTestServer(t, Config{}, svc, fakeDecoder{})
			for path, want := range map[string]string{"/asr": c.asr, "/detect-language": c.detect} {
				resp, body := post(t, srv.URL+path, wavUpload(t))
				if resp.StatusCode != c.status || body != want {
					t.Errorf("%s: %d %s, want %d %s", path, resp.StatusCode, body, c.status, want)
				}
				// The audio was decoded, so the work is known even on failure.
				if d := resp.Header.Get(durationHeader); d != "1.428" {
					t.Errorf("%s: duration header %q", path, d)
				}
				if strings.Contains(body, "secret") {
					t.Errorf("%s: leaked cause: %s", path, body)
				}
			}
		})
	}
}

func TestDetectUnusableResult(t *testing.T) {
	for name, det := range map[string]*core.Detection{
		"unknown language": {Language: "xx", Confidence: 0.5},
		"nan confidence":   {Language: "en", Confidence: math.NaN()},
	} {
		t.Run(name, func(t *testing.T) {
			svc := newFakeService()
			svc.detection = det
			srv := newTestServer(t, Config{}, svc, fakeDecoder{})
			resp, body := post(t, srv.URL+"/detect-language", wavUpload(t))
			if resp.StatusCode != 500 || body != "Internal Server Error" {
				t.Errorf("got %d %q", resp.StatusCode, body)
			}
		})
	}
}

func TestDetectJSONFloats(t *testing.T) {
	svc := newFakeService()
	svc.detection = &core.Detection{Language: "de", Confidence: 1}
	srv := newTestServer(t, Config{}, svc, fakeDecoder{})
	pcm := make([]byte, 2*16000*2)
	_, body := post(t, srv.URL+"/detect-language?encode=false", multipartBody(filePart("audio_file", "a", pcm)))
	// Python prints floats with a ".0"; Go's encoding/json would not.
	want := `{"detected_language":"german","language_code":"de","confidence":1.0,"audio_duration_seconds":2.0}`
	if body != want {
		t.Errorf("body %s, want %s", body, want)
	}
	if r := svc.detects[0]; r.Endpoint != "/detect-language" || r.Protocol != core.ProtocolLegacy {
		t.Errorf("detect request %+v", r)
	}
}

func TestMaxUploadBytes(t *testing.T) {
	svc := newFakeService()
	srv := newTestServer(t, Config{MaxUploadBytes: 1000}, svc, fakeDecoder{})
	resp, body := post(t, srv.URL+"/asr", wavUpload(t))
	if resp.StatusCode != 413 || body != `{"detail":"Upload exceeds the maximum size of 1000 bytes"}` {
		t.Errorf("got %d %s", resp.StatusCode, body)
	}
	if len(svc.requests) != 0 {
		t.Errorf("service called")
	}
}

// streamDecoder reports when it sees the first upload bytes.
type streamDecoder struct {
	fakeDecoder
	first chan struct{}
	got   []byte
}

func (d *streamDecoder) DecodeFile(_ context.Context, r io.Reader) (core.PCM, error) {
	buf := make([]byte, 4096)
	n, err := io.ReadFull(r, buf)
	if err != nil {
		return nil, err
	}
	close(d.first)
	rest, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	d.got = append(buf[:n], rest...)
	return core.PCM(make([]byte, 32000)), nil
}

// TestUploadStreamsIntoDecoder sends a body whose tail is withheld until
// the decoder has received data, which is only possible if the handler
// streams the part instead of buffering the whole request. It also checks
// the part bytes arrive intact across parser buffer boundaries, with
// near-delimiter sequences in the data.
func TestUploadStreamsIntoDecoder(t *testing.T) {
	dec := &streamDecoder{first: make(chan struct{})}
	svc := newFakeService()
	srv := newTestServer(t, Config{}, svc, dec)

	data := make([]byte, 300<<10)
	rand.New(rand.NewSource(1)).Read(data)
	for i := 0; i+8 < len(data); i += 9973 {
		copy(data[i:], "\r\n--z\r\n-")
	}
	body := multipartBody(filePart("audio_file", "big.bin", data))
	split := len(body) - 1000

	pr, pw := io.Pipe()
	go func() {
		pw.Write(body[:split])
		select {
		case <-dec.first:
		case <-time.After(5 * time.Second):
			pw.CloseWithError(errors.New("decoder never saw data before the body ended"))
			return
		}
		pw.Write(body[split:])
		pw.Close()
	}()
	resp, err := http.Post(srv.URL+"/asr", mpType, pr)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if !bytes.Equal(dec.got, data) {
		t.Errorf("decoder got %d bytes, want %d identical", len(dec.got), len(data))
	}
}

func TestOpenAPIFollowsDefaultEngine(t *testing.T) {
	cases := []struct {
		model core.Model
		file  string
	}{
		{core.Model{Engine: "faster_whisper"}, "openapi-faster_whisper.json"},
		{core.Model{Engine: "openai_whisper"}, "openapi-openai_whisper.json"},
		{core.Model{Engine: "whisperx"}, "openapi-whisperx.json"},
		{core.Model{Engine: "whisperx", Capabilities: core.Capabilities{Diarize: true}}, "openapi-whisperx_diarize.json"},
		{core.Model{Engine: "fake"}, "openapi-openai_whisper.json"},
	}
	for _, c := range cases {
		svc := newFakeService()
		c.model.ID, c.model.Default = "m", true
		svc.models = []core.Model{{ID: "other", Engine: "faster_whisper"}, c.model}
		srv := newTestServer(t, Config{}, svc, fakeDecoder{})
		resp, err := http.Get(srv.URL + "/openapi.json")
		if err != nil {
			t.Fatal(err)
		}
		got, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		want, err := os.ReadFile("static/" + c.file)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%+v: did not serve %s", c.model, c.file)
		}
		var doc map[string]any
		if err := json.Unmarshal(got, &doc); err != nil {
			t.Errorf("%s: %v", c.file, err)
		}
	}
}

func TestAsrEngineFallsBackToModelEngine(t *testing.T) {
	svc := &engineless{newFakeService()}
	svc.models = []core.Model{{ID: "large", Engine: "whisperx", Default: true}}
	mux := http.NewServeMux()
	New(Config{}, svc, fakeDecoder{}).Register(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, _ := post(t, srv.URL+"/asr", wavUpload(t))
	if got := resp.Header.Get("Asr-Engine"); got != "whisperx" {
		t.Errorf("Asr-Engine %q", got)
	}
}

type engineless struct{ *fakeService }

func (e *engineless) Transcribe(ctx context.Context, req *core.Request, f func(core.Segment) error) (*core.Result, error) {
	res, err := e.fakeService.Transcribe(ctx, req, f)
	if res != nil {
		res.Engine = ""
	}
	return res, err
}

// While no worker has reported a model the engine is unknown, so the
// schema is too: 503 in upstream's error shape, and the right schema as
// soon as a model appears (nothing is cached).
func TestOpenAPIWhileNoWorkerIsUp(t *testing.T) {
	svc := &fakeService{}
	mux := http.NewServeMux()
	New(Config{}, svc, fakeDecoder{}).Register(mux)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/openapi.json", nil))
	if rec.Code != http.StatusServiceUnavailable || !strings.HasPrefix(rec.Body.String(), `{"detail":`) {
		t.Fatalf("no models: %d %s", rec.Code, rec.Body)
	}
	svc.models = []core.Model{{ID: "tiny", Engine: "faster_whisper", Default: true}}
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/openapi.json", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "vad_filter") {
		t.Fatalf("with a faster_whisper model: %d", rec.Code)
	}
}
