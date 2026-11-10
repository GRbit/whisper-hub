package main

// End-to-end tests of the composed gateway: real config loading, real
// ffmpeg decoding, real auth, hub and frontends, with the engine worker
// replaced by the workertest fake over HTTP. Each test binds 127.0.0.1:0
// and talks to the listeners like an external client would.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/auth"
	"github.com/GRbit/whisper-hub/internal/config"
	"github.com/GRbit/whisper-hub/internal/hub/workertest"
)

func needFFmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not in PATH")
	}
}

func threeModels() workertest.Info {
	info := workertest.DefaultInfo()
	info.Models = []workertest.ModelInfo{{ID: "tiny", Default: true}, {ID: "base"}, {ID: "small"}}
	return info
}

type running struct {
	g      *gateway
	cancel context.CancelFunc
	done   chan error
	base   string // http://addr
}

// startGateway loads the configuration exactly as the binary does, from
// environment variables, and runs the gateway until the test ends.
func startGateway(t *testing.T, w *workertest.Worker, env map[string]string) *running {
	t.Helper()
	needFFmpeg(t)
	full := map[string]string{
		"WHISPER_HUB_HTTP_LISTEN":       "127.0.0.1:0",
		"WHISPER_HUB_WYOMING_LISTEN":    "127.0.0.1:0",
		"WHISPER_HUB_HUB_WORKERS":       fmt.Sprintf("[{url: %q}]", w.URL),
		"WHISPER_HUB_HUB_RETRY_MIN":     "20ms",
		"WHISPER_HUB_HUB_RETRY_MAX":     "50ms",
		"WHISPER_HUB_HUB_POLL_INTERVAL": "50ms",
		"WHISPER_HUB_AUDIO_SPOOL_DIR":   t.TempDir(),
	}
	for k, v := range env {
		full[k] = v
	}
	cfg, err := config.Load(nil, func(k string) (string, bool) { v, ok := full[k]; return v, ok }, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	if testing.Verbose() {
		log = newLogger(config.Log{Level: "debug"}, os.Stderr)
	}
	g, err := newGateway(context.Background(), cfg, log)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &running{g: g, cancel: cancel, done: make(chan error, 1), base: "http://" + g.addr(0)}
	go func() { r.done <- g.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-r.done:
		case <-time.After(10 * time.Second):
			t.Error("gateway did not stop")
		}
	})
	waitHealthy(t, r.base)
	return r
}

func waitHealthy(t *testing.T, base string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		resp, err := http.Get(base + "/healthz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("gateway not healthy: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

var wavCache = map[float64][]byte{}

// wav renders a sine tone with ffmpeg, so the upload goes through the real
// decoder.
func wav(t *testing.T, seconds float64) []byte {
	t.Helper()
	if b, ok := wavCache[seconds]; ok {
		return b
	}
	p := filepath.Join(t.TempDir(), "a.wav")
	out, err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i",
		fmt.Sprintf("sine=frequency=440:sample_rate=44100:duration=%g", seconds), "-ac", "2", p).CombinedOutput()
	if err != nil {
		t.Fatalf("ffmpeg: %v %s", err, out)
	}
	b, _ := os.ReadFile(p)
	wavCache[seconds] = b
	return b
}

type cred func(*http.Request)

func bearer(key string) cred {
	return func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+key) }
}
func basic(u, p string) cred { return func(r *http.Request) { r.SetBasicAuth(u, p) } }
func none(*http.Request)     {}

func transcribe(t *testing.T, base string, c cred, model string, audio []byte) (int, map[string]any) {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "a.wav")
	_, _ = fw.Write(audio)
	_ = mw.WriteField("model", model)
	_ = mw.Close()
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	c(req)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func getJSON(t *testing.T, url string, c cred) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	c(req)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func modelIDs(t *testing.T, base string, c cred) []string {
	t.Helper()
	code, out := getJSON(t, base+"/v1/models", c)
	if code != http.StatusOK {
		t.Fatalf("/v1/models: %d %v", code, out)
	}
	var ids []string
	for _, m := range out["data"].([]any) {
		ids = append(ids, m.(map[string]any)["id"].(string))
	}
	slices.Sort(ids)
	return ids
}

// --- minimal Wyoming client ---

type wyEvent struct {
	Type string         `json:"type"`
	Data map[string]any `json:"data,omitempty"`
}

func wySend(t *testing.T, c net.Conn, typ string, data map[string]any, payload []byte) {
	t.Helper()
	h := map[string]any{"type": typ, "version": "1.11.0"}
	var db []byte
	if data != nil {
		db, _ = json.Marshal(data)
		h["data_length"] = len(db)
	}
	if len(payload) > 0 {
		h["payload_length"] = len(payload)
	}
	hb, _ := json.Marshal(h)
	if _, err := c.Write(slices.Concat(hb, []byte("\n"), db, payload)); err != nil {
		t.Fatal(err)
	}
}

func wyRead(t *testing.T, r *bufio.Reader) wyEvent {
	t.Helper()
	line, err := r.ReadBytes('\n')
	if err != nil {
		t.Fatalf("wyoming read: %v", err)
	}
	var h struct {
		wyEvent
		DataLength    int `json:"data_length"`
		PayloadLength int `json:"payload_length"`
	}
	if err := json.Unmarshal(line, &h); err != nil {
		t.Fatal(err)
	}
	if h.DataLength > 0 {
		b := make([]byte, h.DataLength)
		if _, err := io.ReadFull(r, b); err != nil {
			t.Fatal(err)
		}
		if h.Data == nil {
			h.Data = map[string]any{}
		}
		_ = json.Unmarshal(b, &h.Data)
	}
	_, _ = io.CopyN(io.Discard, r, int64(h.PayloadLength))
	return h.wyEvent
}

// wyTranscribe sends one utterance of 16 kHz mono silence and returns the
// final event (transcript or error).
func wyTranscribe(t *testing.T, addr string, seconds float64) wyEvent {
	t.Helper()
	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	_ = c.SetDeadline(time.Now().Add(10 * time.Second))
	r := bufio.NewReader(c)
	fmtData := map[string]any{"rate": 16000, "width": 2, "channels": 1}
	wySend(t, c, "transcribe", map[string]any{"language": "en"}, nil)
	wySend(t, c, "audio-start", fmtData, nil)
	wySend(t, c, "audio-chunk", fmtData, make([]byte, int(seconds*16000)*2))
	wySend(t, c, "audio-stop", nil, nil)
	for {
		ev := wyRead(t, r)
		if ev.Type == "transcript" || ev.Type == "error" {
			return ev
		}
	}
}

func wyDescribe(t *testing.T, addr string) wyEvent {
	t.Helper()
	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	_ = c.SetDeadline(time.Now().Add(10 * time.Second))
	wySend(t, c, "describe", nil, nil)
	return wyRead(t, bufio.NewReader(c))
}

func readLog(t *testing.T, path string) []map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	for _, line := range bytes.Split(bytes.TrimSpace(b), []byte("\n")) {
		var m map[string]any
		if err := json.Unmarshal(line, &m); err != nil {
			t.Fatalf("log line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func hasLine(lines []map[string]any, kv ...string) bool {
	for _, l := range lines {
		ok := true
		for i := 0; i < len(kv); i += 2 {
			if fmt.Sprint(l[kv[i]]) != kv[i+1] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// --- tests ---

func TestAuthOffOpenAIAndWyoming(t *testing.T) {
	w := workertest.New(t, threeModels())
	logFile := filepath.Join(t.TempDir(), "requests.jsonl")
	r := startGateway(t, w, map[string]string{"WHISPER_HUB_AUTH_USAGE_LOG_FILE": logFile})

	code, out := transcribe(t, r.base, none, "whisper-1", wav(t, 2))
	if code != http.StatusOK || !strings.Contains(fmt.Sprint(out["text"]), "s0 s1") {
		t.Fatalf("transcription: %d %v", code, out)
	}
	job := w.Jobs()[0]
	if job.Model != "tiny" || len(job.Audio) != 2*16000*2 {
		t.Fatalf("worker job model %q audio %d bytes", job.Model, len(job.Audio))
	}
	if ids := modelIDs(t, r.base, none); !slices.Equal(ids, []string{"base", "small", "tiny", "whisper-1"}) {
		t.Fatalf("models %v", ids)
	}

	info := wyDescribe(t, r.g.wyAddr())
	if info.Type != "info" || !strings.Contains(fmt.Sprint(info.Data["asr"]), "tiny") {
		t.Fatalf("describe: %+v", info)
	}
	ev := wyTranscribe(t, r.g.wyAddr(), 1)
	if ev.Type != "transcript" || !strings.Contains(fmt.Sprint(ev.Data["text"]), "s0") {
		t.Fatalf("wyoming: %+v", ev)
	}
	// Usage endpoint has no admin to check while auth is off.
	if code, _ := getJSON(t, r.base+auth.UsagePath, none); code != http.StatusForbidden {
		t.Fatalf("usage endpoint with auth off: %d", code)
	}
	lines := readLog(t, logFile)
	if !hasLine(lines, "user", "anonymous", "protocol", "openai", "status", "ok", "model", "tiny") ||
		!hasLine(lines, "user", "anonymous", "protocol", "wyoming", "status", "ok") {
		t.Fatalf("usage log %v", lines)
	}
}

func TestAuthOnAccessControlAndUsage(t *testing.T) {
	w := workertest.New(t, threeModels())
	dir := t.TempDir()
	aliceKey, aliceHash, _ := auth.NewAPIKey()
	carolKey, carolHash, _ := auth.NewAPIKey()
	bobHash, err := auth.HashPassword("bob-secret")
	if err != nil {
		t.Fatal(err)
	}
	users := filepath.Join(dir, "users.yaml")
	if err := os.WriteFile(users, []byte(fmt.Sprintf(`users:
  - name: alice
    api_keys: [%q]
    models: [base]
  - name: bob
    password_hash: %q
  - name: carol
    api_keys: [%q]
    admin: true
`, aliceHash, bobHash, carolHash)), 0o600); err != nil {
		t.Fatal(err)
	}
	logFile := filepath.Join(dir, "requests.jsonl")
	r := startGateway(t, w, map[string]string{
		"WHISPER_HUB_AUTH_USERS_FILE":     users,
		"WHISPER_HUB_AUTH_USAGE_LOG_FILE": logFile,
		"WHISPER_HUB_AUTH_WYOMING_USER":   "bob",
	})
	alice, bob, carol := bearer(aliceKey), basic("bob", "bob-secret"), bearer(carolKey)

	code, out := transcribe(t, r.base, none, "", wav(t, 1))
	if code != http.StatusUnauthorized || out["error"] == nil {
		t.Fatalf("no credentials: %d %v", code, out)
	}
	if code, _ := transcribe(t, r.base, bearer("wh_wrong"), "", wav(t, 1)); code != http.StatusUnauthorized {
		t.Fatalf("wrong key: %d", code)
	}
	// /healthz stays public.
	if code, _ := getJSON(t, r.base+"/healthz", none); code != http.StatusOK {
		t.Fatalf("healthz: %d", code)
	}

	if ids := modelIDs(t, r.base, alice); !slices.Equal(ids, []string{"base", "tiny", "whisper-1"}) {
		t.Fatalf("alice models %v", ids)
	}
	if ids := modelIDs(t, r.base, bob); !slices.Equal(ids, []string{"tiny", "whisper-1"}) {
		t.Fatalf("bob models %v", ids)
	}
	if code, out := transcribe(t, r.base, alice, "base", wav(t, 1)); code != http.StatusOK {
		t.Fatalf("alice base: %d %v", code, out)
	}
	code, out = transcribe(t, r.base, bob, "base", wav(t, 1))
	if code != http.StatusForbidden || out["error"] == nil {
		t.Fatalf("bob base: %d %v", code, out)
	}
	if code, _ := transcribe(t, r.base, bob, "whisper-1", wav(t, 1)); code != http.StatusOK {
		t.Fatalf("bob whisper-1: %d", code)
	}
	if ev := wyTranscribe(t, r.g.wyAddr(), 1); ev.Type != "transcript" {
		t.Fatalf("wyoming as bob: %+v", ev)
	}

	if code, _ := getJSON(t, r.base+auth.UsagePath, alice); code != http.StatusForbidden {
		t.Fatalf("non-admin usage: %d", code)
	}
	code, out = getJSON(t, r.base+auth.UsagePath, carol)
	if code != http.StatusOK {
		t.Fatalf("admin usage: %d %v", code, out)
	}
	us := out["users"].(map[string]any)
	if fmt.Sprint(us["alice"].(map[string]any)["ok_requests"]) != "1" ||
		fmt.Sprint(us["bob"].(map[string]any)["requests"]) != "3" ||
		fmt.Sprint(us["bob"].(map[string]any)["failed_requests"]) != "1" {
		t.Fatalf("totals %v", us)
	}
	lines := readLog(t, logFile)
	for _, want := range [][]string{
		{"user", "alice", "model", "base", "status", "ok"},
		{"user", "bob", "model", "base", "status", "forbidden"},
		{"user", "bob", "protocol", "wyoming", "status", "ok"},
	} {
		if !hasLine(lines, want...) {
			t.Errorf("usage log lacks %v", want)
		}
	}
}

func TestWyomingSourceAllowlist(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	r := startGateway(t, w, map[string]string{"WHISPER_HUB_AUTH_WYOMING_ALLOW_CIDRS": "203.0.113.0/24"})
	ev := wyDescribe(t, r.g.wyAddr())
	if ev.Type != "error" || ev.Data["code"] != "forbidden" {
		t.Fatalf("got %+v", ev)
	}
	// Home Assistant writes its whole utterance before reading. The error
	// event must still arrive: closing with the client's events unread
	// makes the kernel send RST, which discards it on the client side.
	for range 20 {
		c, err := net.Dial("tcp", r.g.wyAddr())
		if err != nil {
			t.Fatal(err)
		}
		_ = c.SetDeadline(time.Now().Add(5 * time.Second))
		fmtData := map[string]any{"rate": 16000, "width": 2, "channels": 1}
		wySend(t, c, "transcribe", map[string]any{"language": "en"}, nil)
		wySend(t, c, "audio-start", fmtData, nil)
		time.Sleep(20 * time.Millisecond) // let the server reject and close meanwhile
		for range 10 {
			_, _ = c.Write(frame("audio-chunk", fmtData, make([]byte, 3200)))
		}
		line, err := bufio.NewReader(c).ReadBytes('\n')
		c.Close()
		if err != nil || !strings.Contains(string(line), `"error"`) {
			t.Fatalf("HA-style client lost the error event: %v %q", err, line)
		}
	}
}

func frame(typ string, data map[string]any, payload []byte) []byte {
	db, _ := json.Marshal(data)
	hb, _ := json.Marshal(map[string]any{"type": typ, "version": "1.11.0", "data_length": len(db), "payload_length": len(payload)})
	return slices.Concat(hb, []byte("\n"), db, payload)
}

func TestHealthzFollowsWorker(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetHealthy(false)
	needFFmpeg(t)
	// Start without waiting for health: the worker is still "loading".
	cfg, err := config.Load(nil, func(k string) (string, bool) {
		v, ok := map[string]string{
			"WHISPER_HUB_HTTP_LISTEN": "127.0.0.1:0", "WHISPER_HUB_WYOMING_ENABLED": "false",
			"WHISPER_HUB_HUB_WORKERS": fmt.Sprintf("[{url: %q}]", w.URL), "WHISPER_HUB_HUB_RETRY_MIN": "20ms",
			"WHISPER_HUB_HUB_RETRY_MAX": "20ms", "WHISPER_HUB_HUB_POLL_INTERVAL": "20ms", "WHISPER_HUB_HUB_FAIL_THRESHOLD": "1",
		}[k]
		return v, ok
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	g, err := newGateway(context.Background(), cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- g.Run(ctx) }()
	defer func() { cancel(); <-done }()
	base := "http://" + g.addr(0)

	code, out := getJSON(t, base+"/healthz", none)
	if code != http.StatusServiceUnavailable || out["status"] != "unavailable" {
		t.Fatalf("healthz while loading: %d %v", code, out)
	}
	if code, _ := transcribe(t, base, none, "", wav(t, 1)); code != http.StatusServiceUnavailable {
		t.Fatalf("transcription while loading: %d", code)
	}
	// Legacy answers in upstream's shape instead of guessing the schema.
	resp := asr(t, base, "/asr", none, wav(t, 1))
	if body := readBody(t, resp); resp.StatusCode != http.StatusServiceUnavailable || !strings.HasPrefix(body, `{"detail":`) {
		t.Fatalf("/asr while loading: %d %s", resp.StatusCode, body)
	}
	resp, err = http.Get(base + "/openapi.json")
	if err != nil {
		t.Fatal(err)
	}
	if body := readBody(t, resp); resp.StatusCode != http.StatusServiceUnavailable || !strings.HasPrefix(body, `{"detail":`) {
		t.Fatalf("/openapi.json while loading: %d %s", resp.StatusCode, body)
	}
	w.SetHealthy(true)
	waitHealthy(t, base)
	if resp, err := http.Get(base + "/openapi.json"); err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("/openapi.json once up: %v %v", resp, err)
	}
	w.SetHealthy(false)
	deadline := time.Now().Add(5 * time.Second)
	for {
		if code, _ := getJSON(t, base+"/healthz", none); code == http.StatusServiceUnavailable {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("healthz did not notice the worker going down")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// SIGTERM semantics: the listener closes at once, the job in flight
// completes within the grace period and Run returns cleanly.
func TestGracefulShutdownFinishesJobInFlight(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	release := make(chan struct{})
	w.SetTranscribe(workertest.Block(release))
	r := startGateway(t, w, map[string]string{"WHISPER_HUB_HTTP_SHUTDOWN_GRACE": "10s"})
	audio := wav(t, 1)
	result := make(chan int, 1)
	go func() {
		code, _ := transcribe(t, r.base, none, "", audio)
		result <- code
	}()
	<-w.Started()
	r.cancel()
	deadline := time.Now().Add(5 * time.Second)
	for {
		c, err := net.DialTimeout("tcp", r.g.addr(0), 200*time.Millisecond)
		if err != nil {
			break // no longer accepting
		}
		c.Close()
		if time.Now().After(deadline) {
			t.Fatal("listener still accepting after shutdown started")
		}
		time.Sleep(10 * time.Millisecond)
	}
	select {
	case err := <-r.done:
		t.Fatalf("Run returned with a job in flight: %v", err)
	default:
	}
	close(release)
	if code := <-result; code != http.StatusOK {
		t.Fatalf("in-flight job got %d", code)
	}
	select {
	case err := <-r.done:
		r.done <- err // for the cleanup
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	// A hang bound, not a speed check: http.Server.Shutdown waits up to 5 s
	// for connections that never sent a request, and Go's client transport
	// leaves such spare connections in its pool, so 5 s here is too tight
	// for a loaded machine.
	case <-time.After(20 * time.Second):
		t.Fatal("Run did not return after the job finished")
	}
}

// When the grace period runs out the request is cut off and the worker
// job canceled, instead of shutdown hanging on an hour-long file.
func TestShutdownGraceCutsOffLongJob(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(workertest.Block(make(chan struct{})))
	r := startGateway(t, w, map[string]string{"WHISPER_HUB_HTTP_SHUTDOWN_GRACE": "200ms"})
	audio := wav(t, 1)
	go transcribeQuiet(r.base, audio)
	job := <-w.Started()
	start := time.Now()
	r.cancel()
	select {
	case err := <-r.done:
		r.done <- err
	case <-time.After(5 * time.Second):
		t.Fatal("Run hung past the grace period")
	}
	if el := time.Since(start); el > 3*time.Second {
		t.Fatalf("shutdown took %v", el)
	}
	select {
	case <-job.Ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("worker job not canceled")
	}
}

func transcribeQuiet(base string, audio []byte) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "a.wav")
	_, _ = fw.Write(audio)
	_ = mw.Close()
	resp, err := http.Post(base+"/v1/audio/transcriptions", mw.FormDataContentType(), &body)
	if err == nil {
		resp.Body.Close()
	}
}

func TestStartupErrors(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg, err := config.Load(nil, func(k string) (string, bool) {
		v, ok := map[string]string{"WHISPER_HUB_HTTP_LISTEN": "127.0.0.1:0", "WHISPER_HUB_WYOMING_LISTEN": "127.0.0.1:0",
			"WHISPER_HUB_HUB_WORKERS": fmt.Sprintf("[{url: %q}]", w.URL)}[k]
		return v, ok
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	bad := cfg
	bad.Audio.FFmpegPath = "/nonexistent/ffmpeg"
	if _, err := newGateway(context.Background(), bad, log); err == nil || !strings.Contains(err.Error(), "ffmpeg") {
		t.Fatalf("missing ffmpeg: %v", err)
	}
	needFFmpeg(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	busy := cfg
	busy.HTTP.Listen = ln.Addr().String()
	if _, err := newGateway(context.Background(), busy, log); err == nil || !strings.Contains(err.Error(), "listen") {
		t.Fatalf("port in use: %v", err)
	}
}

// --- legacy (whisper-asr-webservice) API ---

func asr(t *testing.T, base, query string, c cred, audio []byte) *http.Response {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("audio_file", "a.wav")
	_, _ = fw.Write(audio)
	_ = mw.Close()
	req, _ := http.NewRequest(http.MethodPost, base+query, &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	c(req)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func readBody(t *testing.T, resp *http.Response) string {
	t.Helper()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The body of /asr is what the worker rendered (the fake renders
// "<format>:<text>"); everything around it comes from the legacy frontend.
func TestLegacyEveryOutputFormat(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	r := startGateway(t, w, nil)
	audio := wav(t, 2)
	for _, format := range []string{"txt", "vtt", "srt", "tsv", "json"} {
		resp := asr(t, r.base, "/asr?output="+format, none, audio)
		body := readBody(t, resp)
		if resp.StatusCode != http.StatusOK || body != format+": s0 s1" {
			t.Fatalf("%s: %d %q", format, resp.StatusCode, body)
		}
		if resp.Header.Get("Asr-Engine") != "faster_whisper" ||
			resp.Header.Get("Content-Disposition") != `attachment; filename="a.wav.`+format+`"` {
			t.Fatalf("%s headers %v", format, resp.Header)
		}
		job := w.Jobs()[len(w.Jobs())-1]
		if job.Options.LegacyOutput == nil || *job.Options.LegacyOutput != format || len(job.Audio) != 2*16000*2 {
			t.Fatalf("%s: worker options %s, %d audio bytes", format, job.RawOptions, len(job.Audio))
		}
	}
	// The default output is txt.
	if body := readBody(t, asr(t, r.base, "/asr", none, audio)); body != "txt: s0 s1" {
		t.Fatalf("default output %q", body)
	}

	resp := asr(t, r.base, "/detect-language", none, audio)
	var det map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&det); err != nil || resp.StatusCode != http.StatusOK ||
		det["detected_language"] != "english" || det["language_code"] != "en" {
		t.Fatalf("detect-language: %d %v %v", resp.StatusCode, det, err)
	}

	// Upstream's decode error, through the real ffmpeg.
	resp = asr(t, r.base, "/asr", none, []byte("not audio"))
	if body := readBody(t, resp); resp.StatusCode != http.StatusBadRequest ||
		body != `{"detail":"Failed to decode audio (9 bytes, magic=6e6f7420617564696f)"}` {
		t.Fatalf("bad audio: %d %s", resp.StatusCode, body)
	}
	// The catch-all is legacy's, the specific routes of the others still win.
	noRedirect := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := noRedirect.Get(r.base + "/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusTemporaryRedirect || resp.Header.Get("Location") != "/docs" {
		t.Fatalf("/: %d %v", resp.StatusCode, resp.Header)
	}
	resp, _ = http.Get(r.base + "/nope")
	if body := readBody(t, resp); resp.StatusCode != http.StatusNotFound || body != `{"detail":"Not Found"}` {
		t.Fatalf("/nope: %d %s", resp.StatusCode, body)
	}
	if code, _ := getJSON(t, r.base+"/healthz", none); code != http.StatusOK {
		t.Fatalf("healthz shadowed: %d", code)
	}
}

func TestLegacyAuthBasicAndByteIdentical401(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	hash, err := auth.HashPassword("pw")
	if err != nil {
		t.Fatal(err)
	}
	users := filepath.Join(t.TempDir(), "users.yaml")
	if err := os.WriteFile(users, []byte(fmt.Sprintf("users:\n  - name: bob\n    password_hash: %q\n", hash)), 0o600); err != nil {
		t.Fatal(err)
	}
	r := startGateway(t, w, map[string]string{"WHISPER_HUB_AUTH_USERS_FILE": users})
	audio := wav(t, 1)

	// Exactly what the old sidecar sent (http.Error after setting the
	// Basic challenge), so legacy clients and browsers behave the same.
	for _, c := range []cred{none, basic("bob", "wrong")} {
		resp := asr(t, r.base, "/asr", c, audio)
		body := readBody(t, resp)
		if resp.StatusCode != http.StatusUnauthorized || body != "unauthorized\n" ||
			resp.Header.Get("WWW-Authenticate") != `Basic realm="whisper", charset="UTF-8"` ||
			resp.Header.Get("Content-Type") != "text/plain; charset=utf-8" ||
			resp.Header.Get("X-Content-Type-Options") != "nosniff" {
			t.Fatalf("401: %d %q %v", resp.StatusCode, body, resp.Header)
		}
	}
	resp := asr(t, r.base, "/asr?output=srt", basic("bob", "pw"), audio)
	if body := readBody(t, resp); resp.StatusCode != http.StatusOK || body != "srt: s0" {
		t.Fatalf("authenticated: %d %q", resp.StatusCode, body)
	}
	if resp := asr(t, r.base, "/detect-language", basic("bob", "pw"), audio); resp.StatusCode != http.StatusOK {
		t.Fatalf("detect-language: %d", resp.StatusCode)
	}
	// Documentation stays public.
	if resp, err := http.Get(r.base + "/docs"); err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("/docs: %v %v", resp, err)
	}
}

// All-in-one supervision: the gateway exits when its worker dies (so the
// container restarts), and stops the worker when it shuts down itself.
func TestManagedWorker(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	r := startGateway(t, w, map[string]string{"WHISPER_HUB_WORKER_COMMAND": `[sh, -c, "sleep 0.5; exit 7"]`})
	select {
	case err := <-r.done:
		r.done <- err
		if err == nil || !strings.Contains(err.Error(), "engine worker exited") {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("gateway kept running without its worker")
	}

	pidFile := filepath.Join(t.TempDir(), "pid")
	r = startGateway(t, w, map[string]string{"WHISPER_HUB_WORKER_COMMAND": fmt.Sprintf(`[sh, -c, "echo $$ > %s; exec sleep 30"]`, pidFile)})
	var pid int
	deadline := time.Now().Add(5 * time.Second)
	for pid == 0 && time.Now().Before(deadline) {
		b, _ := os.ReadFile(pidFile)
		fmt.Sscan(string(b), &pid)
		time.Sleep(10 * time.Millisecond)
	}
	r.cancel()
	select {
	case err := <-r.done:
		r.done <- err
	case <-time.After(10 * time.Second):
		t.Fatal("shutdown hung on the worker")
	}
	if pid == 0 || exec.Command("kill", "-0", fmt.Sprint(pid)).Run() == nil {
		t.Fatalf("worker %d still alive after shutdown", pid)
	}

	cfgErr := map[string]string{"WHISPER_HUB_WORKER_COMMAND": "[true]", "WHISPER_HUB_WORKER_ENGINE": "faster_whisper"}
	t.Setenv("ASR_ENGINE", "openai_whisper")
	env := map[string]string{"WHISPER_HUB_HTTP_LISTEN": "127.0.0.1:0", "WHISPER_HUB_WYOMING_ENABLED": "false",
		"WHISPER_HUB_HUB_WORKERS": fmt.Sprintf("[{url: %q}]", w.URL)}
	for k, v := range cfgErr {
		env[k] = v
	}
	cfg, err := config.Load(nil, func(k string) (string, bool) { v, ok := env[k]; return v, ok }, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	_, err = newGateway(context.Background(), cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err == nil || !strings.Contains(err.Error(), "whisper-hub:openai_whisper") {
		t.Fatalf("engine mismatch: %v", err)
	}
}
