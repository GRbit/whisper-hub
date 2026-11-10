package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

func usageReq(user, password string) *http.Request {
	r := httptest.NewRequest(http.MethodGet, UsagePath, nil)
	if user != "" {
		r.SetBasicAuth(user, password)
	}
	return r
}

type usageBody struct {
	Users map[string]UserTotals `json:"users"`
}

func TestUsageHandler(t *testing.T) {
	a := newUsersAuth(t)
	a.Record(context.Background(), okEvent("bob", core.ProtocolLegacy, "tiny", 4))
	a.Record(context.Background(), okEvent("alice", core.ProtocolOpenAI, "small", 1))
	h := a.UsageHandler()

	if w := serve(h, usageReq("", "")); w.Code != http.StatusUnauthorized || w.Header().Get("WWW-Authenticate") == "" {
		t.Errorf("no credentials: %d, want 401 with a Basic challenge", w.Code)
	}
	if w := serve(h, usageReq("bob", "hunter2")); w.Code != http.StatusForbidden {
		t.Errorf("non-admin: %d", w.Code)
	}
	post := usageReq("alice", "secret")
	post.Method = http.MethodPost
	if w := serve(h, post); w.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST: %d", w.Code)
	}

	w := serve(h, usageReq("alice", "secret"))
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("admin: %d %s", w.Code, w.Body.String())
	}
	var body usageBody
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Users["bob"].AudioSeconds != 4 || body.Users["alice"].Requests != 1 {
		t.Errorf("body = %+v", body)
	}

	r := httptest.NewRequest(http.MethodGet, UsagePath+"?user=bob", nil)
	r.Header.Set("X-Api-Key", aliceKey)
	w = serve(h, r)
	body = usageBody{}
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if len(body.Users) != 1 || body.Users["bob"].Requests != 1 {
		t.Errorf("filtered, api key auth: %d %+v", w.Code, body)
	}
}

// Admin access by network gets the usage too: every request while
// authentication is off.
func TestUsageHandlerWithAuthOff(t *testing.T) {
	a := newAuth(t, Config{})
	a.Record(context.Background(), okEvent("", core.ProtocolWyoming, "tiny", 2))
	w := serve(a.UsageHandler(), usageReq("", ""))
	var body usageBody
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != http.StatusOK || body.Users[core.NetworkAdminUser].ByProtocol[core.ProtocolWyoming].Requests != 1 {
		t.Errorf("auth off: %d %s", w.Code, w.Body.String())
	}
}

// With authentication on, the admin port (http.open_listen) serves usage
// without credentials, while the authenticated listener still demands an
// admin user, whatever protected_paths says.
func TestUsageHandlerOnAdminPort(t *testing.T) {
	a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1,
		RateLimit: RateLimitConfig{Failures: -1}, ProtectedPaths: []string{"/v1/"}})
	mux := http.NewServeMux()
	mux.Handle(UsagePath, a.UsageHandler())
	if w := serve(a.AdminPort(mux), usageReq("", "")); w.Code != http.StatusOK {
		t.Errorf("admin port: %d %s", w.Code, w.Body.String())
	}
	main := a.Middleware(mux)
	if w := serve(main, usageReq("", "")); w.Code != http.StatusUnauthorized {
		t.Errorf("main listener, unprotected path, no credentials: %d", w.Code)
	}
	if w := serve(main, usageReq("bob", "hunter2")); w.Code != http.StatusForbidden {
		t.Errorf("main listener, non-admin: %d", w.Code)
	}
}
