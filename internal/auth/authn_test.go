package auth

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/bcrypt"
)

func TestCredentials(t *testing.T) {
	a := newUsersAuth(t)
	xkey := func(k string) *http.Request {
		r := httptest.NewRequest(http.MethodPost, "/v1/models", nil)
		r.Header.Set("X-Api-Key", k)
		return r
	}
	withAuthz := func(v string) *http.Request {
		r := httptest.NewRequest(http.MethodPost, "/asr", nil)
		r.Header.Set("Authorization", v)
		return r
	}
	cases := []struct {
		name string
		req  *http.Request
		want string // "" = rejected
	}{
		{"basic correct", basicReq("alice", "secret"), "alice"},
		{"basic second user", basicReq("bob", "hunter2"), "bob"},
		{"basic wrong password", basicReq("alice", "hunter2"), ""},
		{"basic unknown user", basicReq("dave", "secret"), ""},
		{"basic empty password", basicReq("alice", ""), ""},
		{"basic for key-only user", basicReq("carol", ""), ""},
		{"basic for user without credentials", basicReq("ha", ""), ""},
		{"basic with an api key as password", basicReq("alice", aliceKey), ""},
		{"basic malformed", withAuthz("Basic !!!notbase64"), ""},
		{"bearer correct", keyReq(aliceKey), "alice"},
		{"bearer scheme is case-insensitive", withAuthz("bearer " + aliceKey), "alice"},
		{"bearer on a legacy path", withAuthz("Bearer " + aliceKey), "alice"},
		{"bearer wrong key", keyReq("wh_nope"), ""},
		{"bearer empty", keyReq(""), ""},
		{"bearer disabled user", keyReq(carolKey), ""},
		{"bearer with the stored hash itself", keyReq(keyHash(aliceKey)), ""},
		{"x-api-key correct", xkey(aliceKey), "alice"},
		{"x-api-key wrong", xkey("wh_nope"), ""},
		{"unsupported scheme", withAuthz("Digest foo"), ""},
		{"no credentials", httptest.NewRequest(http.MethodPost, "/asr", nil), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := userOf(a, tc.req); got != tc.want {
				t.Errorf("user = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestDisabledUserWithPasswordIsRejected(t *testing.T) {
	path := writeFile(t, "users:\n  - name: eve\n    disabled: true\n    password_hash: \""+hashOf(t, "pw")+"\"\n")
	a := newAuth(t, Config{UsersFile: path, PollInterval: -1})
	if u := userOf(a, basicReq("eve", "pw")); u != "" {
		t.Error("disabled user was accepted")
	}
}

func TestVerifiedPasswordIsCached(t *testing.T) {
	// The cache is what keeps a 250ms bcrypt off every upload; its hits are
	// visible in the Debug line.
	logs := captureLogs(t)
	a := newUsersAuth(t)
	userOf(a, basicReq("alice", "secret"))
	userOf(a, basicReq("alice", "secret"))
	out := logs.String()
	if strings.Count(out, "cached=false") != 1 || strings.Count(out, "cached=true") != 1 {
		t.Errorf("want one bcrypt check then one cache hit:\n%s", out)
	}
	// A rejected password must never be remembered as valid.
	for range 2 {
		if u := userOf(a, basicReq("alice", "wrong")); u != "" {
			t.Fatal("wrong password accepted")
		}
	}
}

func TestUnknownUserCostsAsMuchAsAWrongPassword(t *testing.T) {
	// No user enumeration through timing: an unknown name runs a dummy
	// bcrypt at the file's cost. Cost 10 (~60ms) dominates noise; the check
	// is a loose factor of two, not a benchmark.
	if testing.Short() {
		t.Skip("bcrypt timing")
	}
	h, err := bcrypt.GenerateFromPassword([]byte("pw"), 10)
	if err != nil {
		t.Fatal(err)
	}
	a := newAuth(t, Config{UsersFile: writeFile(t, "users:\n  - name: alice\n    password_hash: \""+string(h)+"\"\n"),
		PollInterval: -1, RateLimit: RateLimitConfig{Failures: -1}})
	userOf(a, basicReq("nobody", "x")) // the dummy hash is generated on first use

	measure := func(name string) time.Duration {
		best := time.Hour
		for range 3 {
			start := time.Now()
			userOf(a, basicReq(name, "wrong"))
			best = min(best, time.Since(start))
		}
		return best
	}
	known, unknown := measure("alice"), measure("nobody")
	if unknown < known/2 {
		t.Errorf("unknown user took %v, known user %v: timing reveals which logins exist", unknown, known)
	}
}

// limited reports whether the response is a rate limit rejection; plain
// 401s carry no Retry-After.
func limited(w *httptest.ResponseRecorder) bool {
	return w.Code == http.StatusUnauthorized && w.Header().Get("Retry-After") != ""
}

func TestRateLimitBlocksAddressAfterFailures(t *testing.T) {
	const window = 300 * time.Millisecond
	logs := captureLogs(t)
	a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		RateLimit: RateLimitConfig{Failures: 3, Window: window}})
	h := a.Middleware(echoIdentity)

	// Put alice's password in the cache from another address.
	serve(h, from(basicReq("alice", "secret"), "10.0.0.2"))
	for i := range 3 {
		if w := serve(h, from(basicReq("alice", "wrong"), "10.0.0.1")); w.Code != 401 || limited(w) {
			t.Fatalf("failure %d: status %d, limited %v", i, w.Code, limited(w))
		}
	}
	if n := strings.Count(logs.String(), "blocking address"); n != 1 {
		t.Errorf("block announced %d times, want once", n)
	}
	if w := serve(h, from(basicReq("bob", "hunter2"), "10.0.0.1")); !limited(w) {
		t.Errorf("uncached login from a blocked address: %d, want rate limited", w.Code)
	}
	if w := serve(h, from(keyReq("wh_nope"), "10.0.0.1")); !limited(w) {
		t.Errorf("bad key from a blocked address: %d, want rate limited", w.Code)
	}
	// Cheap, unforgeable credentials still pass from a blocked address.
	if w := serve(h, from(basicReq("alice", "secret"), "10.0.0.1")); w.Code != 200 {
		t.Errorf("cached login from a blocked address: %d", w.Code)
	}
	if w := serve(h, from(keyReq(aliceKey), "10.0.0.1")); w.Code != 200 {
		t.Errorf("valid key from a blocked address: %d", w.Code)
	}
	if w := serve(h, from(basicReq("bob", "hunter2"), "10.0.0.3")); w.Code != 200 {
		t.Errorf("other address: %d", w.Code)
	}
	time.Sleep(window + 50*time.Millisecond)
	if w := serve(h, from(basicReq("bob", "hunter2"), "10.0.0.1")); w.Code != 200 {
		t.Errorf("after the window: %d", w.Code)
	}
}

func TestMissingCredentialsDoNotCountAsFailures(t *testing.T) {
	// Browsers and SDK probes send a first request without credentials.
	a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		RateLimit: RateLimitConfig{Failures: 1}})
	h := a.Middleware(echoIdentity)
	for range 5 {
		serve(h, from(httptest.NewRequest(http.MethodPost, "/asr", nil), "10.0.0.1"))
	}
	if w := serve(h, from(basicReq("bob", "hunter2"), "10.0.0.1")); w.Code != 200 {
		t.Errorf("status %d after anonymous probes, want 200", w.Code)
	}
}

func TestForwardedAddressOnlyWhenTrusted(t *testing.T) {
	fwd := func(r *http.Request, client string) *http.Request {
		r.RemoteAddr = "192.0.2.1:5555" // the reverse proxy
		r.Header.Set("X-Forwarded-For", client+", 10.9.9.9")
		return r
	}
	for _, trusted := range []bool{false, true} {
		a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1, TrustForwarded: trusted,
			RateLimit: RateLimitConfig{Failures: 1}})
		h := a.Middleware(echoIdentity)
		serve(h, fwd(keyReq("wh_nope"), "203.0.113.1"))
		w := serve(h, fwd(keyReq("wh_nope"), "203.0.113.2"))
		// Trusted: two clients, two buckets. Untrusted: one proxy address.
		if limited(w) == trusted {
			t.Errorf("trusted=%v: second client limited=%v", trusted, limited(w))
		}
	}
}

func TestRateLimiterMemoryIsBounded(t *testing.T) {
	// Past limiterMaxAddrs tracked addresses new ones are not tracked, so a
	// flood of spoofed sources cannot grow memory; they then simply cost
	// what they would without a limiter. Observable as: a fresh address is
	// not blocked after its first failure.
	if testing.Short() {
		t.Skip("sends 65k requests")
	}
	a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		RateLimit: RateLimitConfig{Failures: 1, Window: time.Hour}})
	h := a.Middleware(echoIdentity)
	for i := range limiterMaxAddrs {
		serve(h, from(keyReq("wh_nope"), "10."+strconv.Itoa(i>>16)+"."+strconv.Itoa(i>>8&255)+"."+strconv.Itoa(i&255)))
	}
	if w := serve(h, from(keyReq("wh_nope"), "10.0.0.0")); !limited(w) {
		t.Fatal("tracked address should be blocked")
	}
	serve(h, from(keyReq("wh_nope"), "172.16.0.1"))
	if w := serve(h, from(keyReq("wh_nope"), "172.16.0.1")); limited(w) {
		t.Error("an address past the cap was tracked")
	}
}
