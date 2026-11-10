package openaiapi_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/openaiapi"
)

const (
	pathTranscriptions = "/v1/audio/transcriptions"
	pathTranslations   = "/v1/audio/translations"
	typeInvalidRequest = "invalid_request_error"
	typeServer         = "server_error"
)

// apiError mirrors the OpenAI error object on the wire.
type apiError struct {
	Message string
	Type    string
	Param   *string
	Code    *string
}

type usage struct {
	Type    string
	Seconds float64
}

type modelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type modelList struct {
	Object string
	Data   []modelObject
}

type fakeDecoder struct {
	calls int
	err   error
}

func (d *fakeDecoder) DecodeFile(ctx context.Context, r io.Reader) (core.PCM, error) {
	d.calls++
	if _, err := io.ReadAll(r); err != nil {
		return nil, core.WrapError(core.KindBadAudio, err, "could not read upload")
	}
	if d.err != nil {
		return nil, d.err
	}
	return make(core.PCM, core.SampleRate*2*2), nil // 2 s
}

func (d *fakeDecoder) FromRaw(ctx context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	return core.PCM(data), nil
}

type fakeService struct {
	mu      sync.Mutex
	req     *core.Request
	err     error
	failAt  int // with onSegment: fail before segment index failAt (>0)
	block   bool
	ctxDone chan struct{}
	models  []core.Model
}

var testSegs = []core.Segment{
	{ID: 0, Start: 0, End: 1, Text: " Hi there.", Tokens: []int{1, 2},
		Words: []core.Word{{Word: " Hi", Start: 0, End: 0.4}, {Word: " there.", Start: 0.4, End: 1}}},
	{ID: 1, Start: 1, End: 3725.5, Text: " Bye.",
		Words: []core.Word{{Word: " Bye.", Start: 1, End: 2}}},
}

func (s *fakeService) Transcribe(ctx context.Context, req *core.Request, onSegment func(core.Segment) error) (*core.Result, error) {
	s.mu.Lock()
	s.req = req
	s.mu.Unlock()
	if s.err != nil && s.failAt == 0 {
		return nil, s.err
	}
	for i, seg := range testSegs {
		if onSegment != nil {
			if s.failAt > 0 && i == s.failAt {
				return nil, s.err
			}
			if err := onSegment(seg); err != nil {
				return nil, err
			}
			if s.block {
				<-ctx.Done()
				close(s.ctxDone)
				return nil, ctx.Err()
			}
		}
	}
	return &core.Result{Model: "tiny", Language: "de", Duration: 2.2, Text: " Hi there. Bye.", Segments: testSegs}, nil
}

func (s *fakeService) DetectLanguage(ctx context.Context, req *core.DetectRequest) (*core.Detection, error) {
	return nil, errors.New("unused")
}

func (s *fakeService) Models(ctx context.Context) []core.Model { return s.models }

func newTest(t *testing.T, cfg openaiapi.Config) (*fakeService, *fakeDecoder, http.Handler) {
	t.Helper()
	svc := &fakeService{models: []core.Model{
		{ID: "tiny", Default: true, Aliases: []string{"whisper-1", "fast"}},
		{ID: "small", Aliases: []string{"whisper-1", "tiny"}},
	}}
	dec := &fakeDecoder{}
	mux := http.NewServeMux()
	openaiapi.New(cfg, svc, dec, slog.New(slog.DiscardHandler)).Register(mux)
	return svc, dec, mux
}

type field struct{ k, v string }

// form builds a multipart body; a field named "file" becomes a file part.
func form(t *testing.T, fields ...field) (*bytes.Buffer, string) {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)
	for _, f := range fields {
		if f.k == "file" {
			fw, _ := mw.CreateFormFile("file", "a.wav")
			_, _ = fw.Write([]byte(f.v))
			continue
		}
		_ = mw.WriteField(f.k, f.v)
	}
	_ = mw.Close()
	return &b, mw.FormDataContentType()
}

func post(t *testing.T, h http.Handler, path string, fields ...field) *httptest.ResponseRecorder {
	t.Helper()
	body, ct := form(t, fields...)
	r := httptest.NewRequest(http.MethodPost, path, body)
	r.Header.Set("Content-Type", ct)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

var audio = field{"file", "RIFFxxxx"}

func decode[T any](t *testing.T, w *httptest.ResponseRecorder) T {
	t.Helper()
	var v T
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatalf("bad json %q: %v", w.Body.String(), err)
	}
	return v
}

func TestJSONDefault(t *testing.T) {
	svc, _, h := newTest(t, openaiapi.Config{})
	w := post(t, h, pathTranscriptions, audio, field{"model", "whisper-1"})
	if w.Code != 200 || w.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status %d ct %q", w.Code, w.Header().Get("Content-Type"))
	}
	want := `{"text":"Hi there. Bye.","usage":{"type":"duration","seconds":3}}`
	if w.Body.String() != want {
		t.Fatalf("got %s want %s", w.Body, want)
	}
	r := svc.req
	if r.Model != "whisper-1" || r.Task != core.TaskTranscribe ||
		r.Protocol != core.ProtocolOpenAI || r.Endpoint != pathTranscriptions || r.WordTimestamps ||
		r.Temperature != nil || r.Language != "" || len(r.Audio) != core.SampleRate*4 {
		t.Fatalf("unexpected request %+v", r)
	}
}

func TestModelAbsentPassedEmpty(t *testing.T) {
	svc, _, h := newTest(t, openaiapi.Config{})
	if w := post(t, h, pathTranscriptions, audio); w.Code != 200 || svc.req.Model != "" {
		t.Fatalf("status %d model %q", w.Code, svc.req.Model)
	}
}

func TestParamsPassThrough(t *testing.T) {
	svc, _, h := newTest(t, openaiapi.Config{})
	w := post(t, h, pathTranscriptions, field{"language", "EN"}, field{"prompt", "names: Bob"},
		field{"temperature", "0.4"}, audio)
	if w.Code != 200 {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
	r := svc.req
	if r.Language != "en" || r.Prompt != "names: Bob" || r.Temperature == nil || *r.Temperature != 0.4 {
		t.Fatalf("unexpected request %+v", r)
	}
	// Temperature 0 means "automatic fallback" in OpenAI terms: engine default.
	post(t, h, pathTranscriptions, field{"temperature", "0"}, audio)
	if svc.req.Temperature != nil {
		t.Fatal("temperature 0 must map to engine default")
	}
}

func TestLanguagesMapping(t *testing.T) {
	svc, _, h := newTest(t, openaiapi.Config{})
	post(t, h, pathTranscriptions, field{"languages[]", "ru"}, audio)
	if svc.req.Language != "ru" {
		t.Fatalf("single languages entry: got %q", svc.req.Language)
	}
	post(t, h, pathTranscriptions, field{"languages[]", "ru"}, field{"languages[]", "de"}, audio)
	if svc.req.Language != "" {
		t.Fatalf("several languages must auto-detect, got %q", svc.req.Language)
	}
	post(t, h, pathTranscriptions, field{"language", "fr"}, field{"languages[]", "ru"}, audio)
	if svc.req.Language != "fr" {
		t.Fatalf("explicit language must win, got %q", svc.req.Language)
	}
}

func TestPlainFormats(t *testing.T) {
	_, _, h := newTest(t, openaiapi.Config{})
	cases := map[string]string{
		"text": "Hi there. Bye.\n",
		"srt":  "1\n00:00:00,000 --> 00:00:01,000\nHi there.\n\n2\n00:00:01,000 --> 01:02:05,500\nBye.\n\n",
		"vtt":  "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi there.\n\n00:00:01.000 --> 01:02:05.500\nBye.\n\n",
	}
	for f, want := range cases {
		w := post(t, h, pathTranscriptions, audio, field{"response_format", f})
		if w.Code != 200 || w.Body.String() != want || w.Header().Get("Content-Type") != "text/plain; charset=utf-8" {
			t.Errorf("%s: status %d ct %q body %q", f, w.Code, w.Header().Get("Content-Type"), w.Body)
		}
	}
}

func TestVerboseJSON(t *testing.T) {
	type resp struct {
		Task     string
		Language string
		Duration float64
		Text     string
		Segments []map[string]any
		Words    []map[string]any
		Usage    *usage
	}
	svc, _, h := newTest(t, openaiapi.Config{})
	w := post(t, h, pathTranscriptions, audio, field{"response_format", "verbose_json"})
	v := decode[resp](t, w)
	if v.Task != "transcribe" || v.Language != "german" || v.Duration != 2.2 || v.Text != "Hi there. Bye." ||
		len(v.Segments) != 2 || v.Words != nil || v.Usage == nil || v.Usage.Seconds != 3 || svc.req.WordTimestamps {
		t.Fatalf("unexpected %+v", v)
	}
	for _, k := range []string{"id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"} {
		if _, ok := v.Segments[0][k]; !ok {
			t.Errorf("segment lacks %s", k)
		}
	}
	if v.Segments[0]["text"] != " Hi there." {
		t.Errorf("segment text must keep Whisper spacing: %q", v.Segments[0]["text"])
	}
	if toks, ok := v.Segments[1]["tokens"].([]any); !ok || len(toks) != 0 {
		t.Errorf("nil tokens must render as [], got %#v", v.Segments[1]["tokens"])
	}

	// Word only: words without segments, as whisper-1 does. The un-bracketed
	// spelling is accepted too.
	w = post(t, h, pathTranscriptions, audio, field{"response_format", "verbose_json"}, field{"timestamp_granularities", "word"})
	v = decode[resp](t, w)
	if !svc.req.WordTimestamps || v.Segments != nil || len(v.Words) != 3 || v.Words[0]["word"] != "Hi" {
		t.Fatalf("word only: %+v", v)
	}
	if _, ok := v.Words[0]["probability"]; ok {
		t.Error("words must not carry probability")
	}
	w = post(t, h, pathTranscriptions, audio, field{"response_format", "verbose_json"},
		field{"timestamp_granularities[]", "word"}, field{"timestamp_granularities[]", "segment"})
	v = decode[resp](t, w)
	if len(v.Segments) != 2 || len(v.Words) != 3 {
		t.Fatalf("word+segment: %+v", v)
	}

	// Words are useless outside verbose_json: do not pay for them.
	post(t, h, pathTranscriptions, audio, field{"timestamp_granularities[]", "word"})
	if svc.req.WordTimestamps {
		t.Fatal("word timestamps requested for json")
	}
}

func TestTranslations(t *testing.T) {
	svc, _, h := newTest(t, openaiapi.Config{})
	w := post(t, h, pathTranslations, audio, field{"language", "de"}, field{"stream", "true"}, field{"model", "x"})
	if w.Code != 200 || w.Body.String() != `{"text":"Hi there. Bye."}` {
		t.Fatalf("status %d body %s", w.Code, w.Body)
	}
	if svc.req.Task != core.TaskTranslate || svc.req.Language != "" || svc.req.Endpoint != pathTranslations || svc.req.Model != "x" {
		t.Fatalf("unexpected request %+v", svc.req)
	}
	w = post(t, h, pathTranslations, audio, field{"response_format", "verbose_json"})
	v := decode[map[string]any](t, w)
	if v["task"] != "translate" || v["language"] != "english" || v["usage"] != nil || v["words"] != nil {
		t.Fatalf("unexpected %v", v)
	}
}

func TestIgnoredGPTFields(t *testing.T) {
	_, _, h := newTest(t, openaiapi.Config{})
	w := post(t, h, pathTranscriptions, audio, field{"include[]", "logprobs"}, field{"chunking_strategy", "auto"},
		field{"chunking_strategy[type]", "server_vad"}, field{"keywords[]", "foo"}, field{"known_speaker_names[]", "agent"})
	if w.Code != 200 {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
}

func errOf(t *testing.T, w *httptest.ResponseRecorder) *apiError {
	t.Helper()
	b := decode[struct{ Error *apiError }](t, w)
	if b.Error == nil {
		t.Fatalf("no error object in %s", w.Body)
	}
	return b.Error
}

func s(p *string) string {
	if p == nil {
		return "<nil>"
	}
	return *p
}

func TestValidationErrors(t *testing.T) {
	cases := []struct {
		name   string
		fields []field
		param  string
		code   string
	}{
		{"bad format", []field{{"response_format", "xml"}, audio}, "response_format", "invalid_value"},
		{"diarized", []field{{"response_format", "diarized_json"}, audio}, "response_format", "unsupported_value"},
		{"bad language", []field{{"language", "xx"}, audio}, "language", "invalid_language_format"},
		{"bad languages", []field{{"languages[]", "zz"}, audio}, "languages", "invalid_language_format"},
		{"bad temperature", []field{{"temperature", "hot"}, audio}, "temperature", "invalid_type"},
		{"temperature range", []field{{"temperature", "1.5"}, audio}, "temperature", "decimal_out_of_range"},
		{"bad granularity", []field{{"timestamp_granularities[]", "char"}, audio}, "timestamp_granularities", "invalid_value"},
		{"bad stream", []field{{"stream", "maybe"}, audio}, "stream", "invalid_type"},
		{"no file", []field{{"model", "whisper-1"}}, "file", "missing_required_parameter"},
	}
	for _, c := range cases {
		_, dec, h := newTest(t, openaiapi.Config{})
		w := post(t, h, pathTranscriptions, c.fields...)
		e := errOf(t, w)
		if w.Code != 400 || e.Type != typeInvalidRequest || s(e.Param) != c.param || s(e.Code) != c.code {
			t.Errorf("%s: status %d err %+v param %s code %s", c.name, w.Code, e, s(e.Param), s(e.Code))
		}
		// Fields sent before the file are validated before paying for decoding.
		if c.name != "no file" && dec.calls != 0 {
			t.Errorf("%s: decoder called on invalid request", c.name)
		}
	}
}

func TestNotMultipart(t *testing.T) {
	_, _, h := newTest(t, openaiapi.Config{})
	r := httptest.NewRequest(http.MethodPost, pathTranscriptions, strings.NewReader(`{"model":"x"}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 400 || errOf(t, w).Type != typeInvalidRequest {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
}

func TestErrorMapping(t *testing.T) {
	cases := []struct {
		kind   core.ErrorKind
		status int
		typ    string
		code   string
	}{
		{core.KindBadRequest, 400, typeInvalidRequest, "<nil>"},
		{core.KindBadAudio, 400, typeInvalidRequest, "invalid_value"},
		{core.KindUnauthenticated, 401, typeInvalidRequest, "invalid_api_key"},
		{core.KindForbidden, 403, typeInvalidRequest, "model_not_allowed"},
		{core.KindModelNotFound, 404, typeInvalidRequest, "model_not_found"},
		{core.KindUnavailable, 503, typeServer, "<nil>"},
		{core.KindInternal, 500, typeServer, "<nil>"},
	}
	for _, c := range cases {
		svc, _, h := newTest(t, openaiapi.Config{})
		svc.err = core.WrapError(c.kind, errors.New("SECRET /srv/path"), "client text")
		w := post(t, h, pathTranscriptions, audio)
		e := errOf(t, w)
		if w.Code != c.status || e.Type != c.typ || s(e.Code) != c.code {
			t.Errorf("%v: status %d %+v code %s", c.kind, w.Code, e, s(e.Code))
		}
		if strings.Contains(w.Body.String(), "SECRET") {
			t.Errorf("%v: cause leaked: %s", c.kind, w.Body)
		}
		if c.kind != core.KindInternal && e.Message != "client text" {
			t.Errorf("%v: message %q", c.kind, e.Message)
		}
	}
	// Plain errors are internal and never shown.
	svc, _, h := newTest(t, openaiapi.Config{})
	svc.err = errors.New("SECRET")
	if w := post(t, h, pathTranscriptions, audio); w.Code != 500 || strings.Contains(w.Body.String(), "SECRET") {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
}

func TestBadAudio(t *testing.T) {
	_, dec, h := newTest(t, openaiapi.Config{})
	dec.err = core.Errorf(core.KindBadAudio, "Audio file might be corrupted or unsupported")
	w := post(t, h, pathTranscriptions, audio)
	if e := errOf(t, w); w.Code != 400 || s(e.Param) != "file" {
		t.Fatalf("status %d %+v", w.Code, e)
	}
}

func TestMaxUpload(t *testing.T) {
	_, _, h := newTest(t, openaiapi.Config{MaxUploadBytes: 1000})
	big := field{"file", strings.Repeat("x", 5000)}
	// Known Content-Length: rejected up front.
	w := post(t, h, pathTranscriptions, big)
	if e := errOf(t, w); w.Code != 413 || !strings.Contains(e.Message, "(1000)") {
		t.Fatalf("status %d %+v", w.Code, e)
	}
	// Chunked upload: caught while streaming into the decoder.
	body, ct := form(t, big)
	r := httptest.NewRequest(http.MethodPost, pathTranscriptions, io.MultiReader(body))
	r.ContentLength = -1
	r.Header.Set("Content-Type", ct)
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 413 {
		t.Fatalf("chunked: status %d %s", w.Code, w.Body)
	}
	// Under the limit passes; unlimited by default.
	if w := post(t, h, pathTranscriptions, audio); w.Code != 200 {
		t.Fatalf("small: status %d", w.Code)
	}
	_, _, h = newTest(t, openaiapi.Config{})
	if w := post(t, h, pathTranscriptions, big); w.Code != 200 {
		t.Fatalf("unlimited: status %d", w.Code)
	}
}

func sseEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, chunk := range strings.Split(strings.TrimSpace(body), "\n\n") {
		data, ok := strings.CutPrefix(chunk, "data: ")
		if !ok {
			t.Fatalf("bad sse chunk %q", chunk)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatalf("bad event %q: %v", data, err)
		}
		out = append(out, m)
	}
	return out
}

func TestStream(t *testing.T) {
	_, _, h := newTest(t, openaiapi.Config{})
	w := post(t, h, pathTranscriptions, audio, field{"stream", "true"})
	if w.Code != 200 || w.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d ct %q", w.Code, w.Header().Get("Content-Type"))
	}
	ev := sseEvents(t, w.Body.String())
	if len(ev) != 3 || ev[0]["type"] != "transcript.text.delta" || ev[0]["delta"] != "Hi there." ||
		ev[1]["delta"] != " Bye." || ev[2]["type"] != "transcript.text.done" || ev[2]["text"] != "Hi there. Bye." {
		t.Fatalf("events %v", ev)
	}
	if u := ev[2]["usage"].(map[string]any); u["type"] != "duration" || u["seconds"] != 3.0 {
		t.Fatalf("usage %v", u)
	}
	// Formats without a text-delta form ignore stream, like whisper-1.
	if w := post(t, h, pathTranscriptions, audio, field{"stream", "true"}, field{"response_format", "srt"}); !strings.HasPrefix(w.Body.String(), "1\n") {
		t.Fatalf("srt with stream: %q", w.Body)
	}
}

func TestStreamErrors(t *testing.T) {
	// Before the first event: a normal HTTP error.
	svc, _, h := newTest(t, openaiapi.Config{})
	svc.err = core.Errorf(core.KindModelNotFound, "nope")
	w := post(t, h, pathTranscriptions, audio, field{"stream", "true"})
	if w.Code != 404 || errOf(t, w).Message != "nope" {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
	// After it: an error event, which the SDK turns into APIError.
	svc, _, h = newTest(t, openaiapi.Config{})
	svc.err = core.Errorf(core.KindUnavailable, "worker gone")
	svc.failAt = 1
	w = post(t, h, pathTranscriptions, audio, field{"stream", "true"})
	ev := sseEvents(t, w.Body.String())
	if len(ev) != 2 || ev[1]["error"].(map[string]any)["message"] != "worker gone" {
		t.Fatalf("events %v", ev)
	}
}

// TestStreamClientDisconnect reads the first event over a real connection,
// hangs up, and waits for the job's context to be canceled; the fake service
// blocks on ctx.Done() after the first segment, so only cancellation frees it.
func TestStreamClientDisconnect(t *testing.T) {
	svc, _, h := newTest(t, openaiapi.Config{})
	svc.block = true
	svc.ctxDone = make(chan struct{})
	srv := httptest.NewServer(h)
	defer srv.Close()

	body, ct := form(t, audio, field{"stream", "true"})
	resp, err := http.Post(srv.URL+pathTranscriptions, ct, body)
	if err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil || !strings.Contains(line, "transcript.text.delta") {
		t.Fatalf("first line %q err %v", line, err)
	}
	_ = resp.Body.Close()
	select {
	case <-svc.ctxDone:
	case <-time.After(5 * time.Second):
		t.Fatal("job not canceled after client disconnect")
	}
}

func TestModels(t *testing.T) {
	_, _, h := newTest(t, openaiapi.Config{})
	get := func(path string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
		return w
	}
	l := decode[modelList](t, get("/v1/models"))
	if l.Object != "list" || l.Data[0].Object != "model" || l.Data[0].OwnedBy != "whisper-hub" || l.Data[0].Created == 0 {
		t.Fatalf("list %+v", l)
	}
	// Real ids first, then aliases; a name shared by two models or equal to
	// an id is listed once.
	var ids []string
	for _, m := range l.Data {
		ids = append(ids, m.ID)
	}
	if got := strings.Join(ids, ","); got != "tiny,small,whisper-1,fast" {
		t.Fatalf("list ids %s", got)
	}
	for _, id := range []string{"small", "whisper-1", "fast"} {
		if w := get("/v1/models/" + id); w.Code != 200 || decode[modelObject](t, w).ID != id {
			t.Fatalf("retrieve %s: %d %s", id, w.Code, w.Body)
		}
	}
	w := get("/v1/models/gpt-9")
	if e := errOf(t, w); w.Code != 404 || s(e.Code) != "model_not_found" {
		t.Fatalf("missing: %d %+v", w.Code, e)
	}
	w = get("/v1/audio/transcriptions")
	if e := errOf(t, w); w.Code != 404 || e.Message != "Invalid URL (GET /v1/audio/transcriptions)" {
		t.Fatalf("wrong method: %d %+v", w.Code, e)
	}
}

func TestWriteError(t *testing.T) {
	w := httptest.NewRecorder()
	openaiapi.WriteError(w, core.Errorf(core.KindUnauthenticated, "Incorrect API key provided."))
	if w.Code != 401 || w.Body.String() != `{"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}` {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
}
