package auth

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"golang.org/x/crypto/bcrypt"
)

// Tests drive the package only through its exported interface: New, the
// middleware, AllowModel, Record/Totals, WyomingIdentity, UsageHandler and
// the files it reads and writes.

func hashOf(t testing.TB, password string) string {
	t.Helper()
	// Cost 4 (the bcrypt minimum) keeps the suite fast; production hashes
	// come from HashPassword at cost 12.
	h, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.MinCost)
	if err != nil {
		t.Fatalf("hash %q: %v", password, err)
	}
	return string(h)
}

// keyHash spells out the users file format for api_keys independently of
// the package code.
func keyHash(key string) string {
	sum := sha256.Sum256([]byte(key))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func writeFile(t testing.TB, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "users.yaml")
	rewrite(t, path, body)
	return path
}

func rewrite(t testing.TB, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write users file: %v", err)
	}
}

// Fixed API keys so tests can present them; real keys come from NewAPIKey.
const (
	aliceKey = "wh_alice-test-key"
	carolKey = "wh_carol-test-key"
)

// validUsers: alice (password + key, models small and large-*, admin), bob
// (password only, no models), carol (key only, disabled), ha (no
// credentials, the Wyoming user).
func validUsers(t testing.TB) string {
	t.Helper()
	return "users:\n" +
		"  - name: alice\n" +
		"    password_hash: \"" + hashOf(t, "secret") + "\"\n" +
		"    api_keys: [\"" + keyHash(aliceKey) + "\"]\n" +
		"    models: [\"small\", \"large-*\"]\n" +
		"    admin: true\n" +
		"  - name: bob\n" +
		"    password_hash: \"" + hashOf(t, "hunter2") + "\"\n" +
		"  - name: carol\n" +
		"    api_keys: [\"" + keyHash(carolKey) + "\"]\n" +
		"    disabled: true\n" +
		"  - name: ha\n" +
		"    models: [\"*\"]\n"
}

func newAuth(t testing.TB, cfg Config) *Auth {
	t.Helper()
	a, err := New(cfg, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

// newUsersAuth turns auth on with validUsers, no polling and no rate limit.
func newUsersAuth(t testing.TB) *Auth {
	t.Helper()
	return newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		RateLimit: RateLimitConfig{Failures: -1}})
}

// echoIdentity reports the caller the middleware stored.
var echoIdentity = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	id := core.IdentityFrom(r.Context())
	w.Header().Set("X-User", id.User)
	if id.Anonymous {
		w.Header().Set("X-Anonymous", "1")
	}
	w.WriteHeader(http.StatusOK)
})

func serve(h http.Handler, r *http.Request) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

// userOf runs r through a's middleware and returns the authenticated user,
// or "" when the request was rejected.
func userOf(a *Auth, r *http.Request) string {
	w := serve(a.Middleware(echoIdentity), r)
	if w.Code != http.StatusOK {
		return ""
	}
	return w.Header().Get("X-User")
}

func basicReq(user, password string) *http.Request {
	r := httptest.NewRequest(http.MethodPost, "/asr", nil)
	r.SetBasicAuth(user, password)
	return r
}

func keyReq(key string) *http.Request {
	r := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", nil)
	r.Header.Set("Authorization", "Bearer "+key)
	return r
}

func from(r *http.Request, addr string) *http.Request {
	r.RemoteAddr = addr + ":40000"
	return r
}

func waitFor(t testing.TB, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

type logLine map[string]any

func readLines(t testing.TB, path string) []logLine {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read request log: %v", err)
	}
	var out []logLine
	sc := bufio.NewScanner(bytes.NewReader(raw))
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}
		var e logLine
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			t.Fatalf("request log line is not valid JSON: %v (%q)", err, sc.Text())
		}
		out = append(out, e)
	}
	return out
}

// syncBuffer collects slog output from several goroutines.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// captureLogs routes slog at Debug level into a buffer for the test's
// duration. Tests using it must not run in parallel.
func captureLogs(t testing.TB) *syncBuffer {
	t.Helper()
	buf := &syncBuffer{}
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return buf
}
