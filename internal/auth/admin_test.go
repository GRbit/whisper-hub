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

func TestUsageHandlerWithAuthOff(t *testing.T) {
	a := newAuth(t, Config{})
	if w := serve(a.UsageHandler(), usageReq("", "")); w.Code != http.StatusForbidden {
		t.Errorf("auth off: %d, want 403", w.Code)
	}
}
