package auth

import (
	"net/http"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

func TestAllowModel(t *testing.T) {
	open := newAuth(t, Config{})
	restricted := newAuth(t, Config{Anonymous: AnonymousConfig{Restrict: true, Models: []string{"base"}}})
	anon := core.IdentityFrom(t.Context())
	user := func(models ...string) *core.Identity { return &core.Identity{User: "u", Models: models} }

	cases := []struct {
		name      string
		a         *Auth
		id        *core.Identity
		model     string
		isDefault bool
		want      bool
	}{
		{"no patterns: default only", open, user(), "tiny", true, true},
		{"no patterns: other denied", open, user(), "small", false, false},
		{"exact id", open, user("small"), "small", false, true},
		{"exact id is not a prefix", open, user("small"), "small.en", false, false},
		{"patterns keep the default", open, user("small"), "tiny", true, true},
		{"trailing glob", open, user("large-*"), "large-v3", false, true},
		{"glob needs its literal part", open, user("large-*"), "large", false, false},
		{"leading glob", open, user("*-v3"), "large-v3", false, true},
		{"glob spans slashes", open, user("Systran/*"), "Systran/faster-whisper-small", false, true},
		{"inner globs", open, user("a*b*c"), "aXXbYYc", false, true},
		{"inner globs miss", open, user("a*b*c"), "aXXbYY", false, false},
		{"second pattern matches", open, user("small", "medium"), "medium", false, true},
		{"star grants all", open, user("*"), "whatever/model", false, true},
		{"anonymous unrestricted", open, anon, "large-v3", false, true},
		{"nil identity is anonymous", open, nil, "large-v3", false, true},
		{"anonymous restricted: default", restricted, anon, "tiny", true, true},
		{"anonymous restricted: granted", restricted, anon, "base", false, true},
		{"anonymous restricted: denied", restricted, anon, "large-v3", false, false},
		{"restriction does not touch users", restricted, user("*"), "large-v3", false, true},
	}
	for _, tc := range cases {
		if got := tc.a.AllowModel(tc.id, tc.model, tc.isDefault); got != tc.want {
			t.Errorf("%s: AllowModel = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestIdentityFromUsersFileDrivesAllowModel(t *testing.T) {
	// End to end: the models list in the file reaches AllowModel through the
	// identity the middleware stores.
	a := newUsersAuth(t)
	var id *core.Identity
	serve(a.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id = core.IdentityFrom(r.Context())
	})), keyReq(aliceKey))
	if !a.AllowModel(id, "large-v3", false) || a.AllowModel(id, "medium", false) {
		t.Errorf("alice (small, large-*) got wrong access with identity %+v", id)
	}
}
