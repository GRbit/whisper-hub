package auth

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

func TestAuthOffPassesEverythingAsAnonymous(t *testing.T) {
	a := newAuth(t, Config{})
	h := a.Middleware(echoIdentity)
	for _, p := range []string{"/asr", "/v1/audio/transcriptions", "/docs"} {
		w := serve(h, httptest.NewRequest(http.MethodPost, p, nil))
		if w.Code != 200 || w.Header().Get("X-Anonymous") != "1" || w.Header().Get("X-User") != "anonymous" {
			t.Errorf("%s: code %d user %q", p, w.Code, w.Header().Get("X-User"))
		}
	}
}

func TestPathPolicy(t *testing.T) {
	h := newUsersAuth(t).Middleware(echoIdentity)
	protected := map[string]bool{
		"/asr": true, "/asr/": true, "/detect-language": true,
		"/v1": true, "/v1/": true, "/v1/audio/transcriptions": true, "/v1/models": true,
		"/v1/../asr": true, "//asr": true,
		"/": false, "/docs": false, "/openapi.json": false, "/assets/app.js": false,
		"/healthz": false, "/asrx": false, "/v10/x": false,
	}
	for p, want := range protected {
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.URL.Path = p
		if got := serve(h, r).Code == http.StatusUnauthorized; got != want {
			t.Errorf("%q protected = %v, want %v", p, got, want)
		}
	}

	custom := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		ProtectedPaths: []string{"/v1/"}}).Middleware(echoIdentity)
	if serve(custom, httptest.NewRequest(http.MethodPost, "/asr", nil)).Code != 200 ||
		serve(custom, httptest.NewRequest(http.MethodGet, "/v1/models", nil)).Code != 401 {
		t.Error("configured protected_paths not honoured")
	}
}

func TestDefaultRejectionIsPlainBasicChallenge(t *testing.T) {
	// http.Error plus a Basic challenge, which is what makes browsers show
	// a login prompt.
	h := newUsersAuth(t).Middleware(echoIdentity)
	for _, r := range []*http.Request{
		httptest.NewRequest(http.MethodPost, "/asr", nil),
		basicReq("alice", "wrong"),
		keyReq("wh_nope"), // no rejecter registered: /v1 gets the default too
	} {
		w := serve(h, r)
		if w.Code != http.StatusUnauthorized {
			t.Errorf("status = %d, want 401", w.Code)
		}
		if got := w.Header().Get("WWW-Authenticate"); got != `Basic realm="whisper", charset="UTF-8"` {
			t.Errorf("WWW-Authenticate = %q", got)
		}
		if got := w.Header().Get("Content-Type"); got != "text/plain; charset=utf-8" {
			t.Errorf("Content-Type = %q", got)
		}
		if w.Body.String() != "unauthorized\n" {
			t.Errorf("body = %q", w.Body.String())
		}
	}
}

func TestRejectersPerPathPrefix(t *testing.T) {
	type call struct {
		who  string
		kind core.ErrorKind
		msg  string
	}
	var calls []call
	writer := func(who string) func(http.ResponseWriter, error) {
		return func(w http.ResponseWriter, err error) {
			calls = append(calls, call{who, core.KindOf(err), core.ClientMessage(err)})
			w.WriteHeader(http.StatusTeapot)
		}
	}
	a, err := New(Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		RateLimit: RateLimitConfig{Failures: 2, Window: time.Minute}},
		map[string]func(http.ResponseWriter, error){
			"/v1/":       writer("openai"),
			"/v1/audio/": writer("audio"),
		})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	h := a.Middleware(echoIdentity)

	serve(h, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	serve(h, from(keyReq("wh_nope"), "10.0.0.1"))
	serve(h, from(keyReq("wh_nope"), "10.0.0.1"))
	w := serve(h, from(keyReq("wh_nope"), "10.0.0.1"))
	if w.Header().Get("Retry-After") != "60" {
		t.Errorf("Retry-After = %q, want 60 on a rate limited rejection", w.Header().Get("Retry-After"))
	}
	if w = serve(h, httptest.NewRequest(http.MethodPost, "/asr", nil)); w.Header().Get("WWW-Authenticate") == "" {
		t.Error("/asr should fall back to the default writer")
	}

	want := []call{
		{"openai", core.KindUnauthenticated, msgMissing},
		{"audio", core.KindUnauthenticated, msgInvalid},
		{"audio", core.KindUnauthenticated, msgInvalid},
		{"audio", core.KindUnauthenticated, msgLimited},
	}
	if len(calls) != len(want) {
		t.Fatalf("calls = %+v, want %+v", calls, want)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Errorf("call %d = %+v, want %+v", i, calls[i], want[i])
		}
	}
}

func TestMiddlewareStoresIdentity(t *testing.T) {
	a := newUsersAuth(t)
	var got *core.Identity
	h := a.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = core.IdentityFrom(r.Context())
	}))
	serve(h, keyReq(aliceKey))
	if got == nil || got.User != "alice" || got.Anonymous || len(got.Models) != 2 {
		t.Fatalf("identity = %+v", got)
	}
	// Public paths stay anonymous even with credentials: no bcrypt there.
	got = nil
	r := httptest.NewRequest(http.MethodGet, "/docs", nil)
	r.SetBasicAuth("alice", "secret")
	serve(h, r)
	if got == nil || !got.Anonymous {
		t.Errorf("public path identity = %+v", got)
	}
}

func TestNoCredentialsInLogs(t *testing.T) {
	// Logs and the request log get shipped to collectors; a password or key
	// (or the base64 of user:password) leaking there has a long tail.
	logs := captureLogs(t)
	logFile := filepath.Join(t.TempDir(), "requests.jsonl")
	a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		Usage: UsageConfig{LogFile: logFile}})
	h := a.Middleware(echoIdentity)

	const password = "sup3rSecretPassw0rd"
	const badKey = "wh_leakedWrongKeyValue"
	for _, r := range []*http.Request{basicReq("alice", password), basicReq("alice", "secret"),
		keyReq(badKey), keyReq(aliceKey)} {
		serve(h, r)
	}
	a.Close()
	reqLog, _ := os.ReadFile(logFile)
	for name, out := range map[string]string{"slog": logs.String(), "request log": string(reqLog)} {
		for _, secret := range []string{password, "secret", badKey, aliceKey, "Basic ", "Bearer "} {
			if strings.Contains(out, secret) {
				t.Errorf("%s contains %q", name, secret)
			}
		}
	}
	if !strings.Contains(logs.String(), "attempted_user=alice") {
		t.Error("a failed login should name the attempted user")
	}
	lines := readLines(t, logFile)
	if len(lines) != 2 {
		t.Fatalf("request log has %d lines, want 2 (the rejections)", len(lines))
	}
	if lines[0]["attempted_user"] != "alice" || lines[0]["status"] != "unauthenticated" || lines[0]["user"] != "" {
		t.Errorf("first rejection = %v", lines[0])
	}
	if lines[1]["protocol"] != core.ProtocolOpenAI || lines[1]["client_ip"] != "192.0.2.1" {
		t.Errorf("key rejection = %v", lines[1])
	}
}

func TestPreflightPassesUnauthenticated(t *testing.T) {
	// Browsers never attach credentials to a CORS preflight.
	h := newUsersAuth(t).Middleware(echoIdentity)
	w := serve(h, httptest.NewRequest(http.MethodOptions, "/v1/audio/transcriptions", nil))
	if w.Code != 200 || w.Header().Get("X-Anonymous") != "1" {
		t.Errorf("OPTIONS: %d anonymous=%q", w.Code, w.Header().Get("X-Anonymous"))
	}
}
