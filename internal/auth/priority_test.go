package auth

import (
	"net/http"
	"strings"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

// A users file priority travels with the identity, for HTTP users and for
// the entry named wyoming; an entry without one leaves it nil so the hub's
// protocol default applies. Zero and negative values are real priorities.
func TestUsersFilePriorityReachesIdentity(t *testing.T) {
	body := validUsers(t) +
		"  - name: wyoming\n    priority: 120\n" +
		"  - name: dave\n    api_keys: [\"" + keyHash("wh_dave-test-key") + "\"]\n    priority: 0\n" +
		"  - name: erin\n    api_keys: [\"" + keyHash("wh_erin-test-key") + "\"]\n    priority: -5\n"
	body = strings.Replace(body, "    admin: true\n", "    admin: true\n    priority: 70\n", 1)
	a := newAuth(t, Config{UsersFile: writeFile(t, body), PollInterval: -1})

	var got *core.Identity
	h := a.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = core.IdentityFrom(r.Context())
	}))
	for key, want := range map[string]*int{aliceKey: ptr(70), "wh_dave-test-key": ptr(0), "wh_erin-test-key": ptr(-5)} {
		got = nil
		serve(h, keyReq(key))
		if got == nil || got.Priority == nil || *got.Priority != *want {
			t.Errorf("key %s: identity %+v, want priority %d", key, got, *want)
		}
	}
	got = nil
	r := basicReq("bob", "hunter2")
	serve(h, r)
	if got == nil || got.User != "bob" || got.Priority != nil {
		t.Errorf("bob: identity %+v, want no priority", got)
	}

	id, err := a.WyomingIdentity(tcp("10.0.0.1"))
	if err != nil || id.Priority == nil || *id.Priority != 120 {
		t.Fatalf("wyoming identity %+v, %v", id, err)
	}
}

func TestUsersFilePriorityMustBeANumber(t *testing.T) {
	_, err := parseUsers([]byte("users:\n  - name: a\n    priority: high\n"))
	if err == nil {
		t.Fatal("a non-numeric priority was accepted")
	}
}

func ptr(n int) *int { return &n }
