package auth

import (
	"net"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

func tcp(ip string) net.Addr { return &net.TCPAddr{IP: net.ParseIP(ip), Port: 40000} }

func TestWyomingIdentity(t *testing.T) {
	users := writeFile(t, validUsers(t))
	cidrs := []string{"192.168.1.0/24", "::1"}

	cases := []struct {
		name      string
		cfg       Config
		remote    net.Addr
		wantUser  string
		wantAdmin bool
		wantKind  core.ErrorKind
		wantErr   bool
	}{
		{"auth off, no user", Config{}, tcp("10.0.0.1"), "admin", true, 0, false},
		{"auth off, named user", Config{Wyoming: WyomingConfig{User: "ha"}}, tcp("10.0.0.1"), "ha", true, 0, false},
		{"auth off, cidr allows", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("192.168.1.7"), "admin", true, 0, false},
		{"auth off, cidr denies", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("10.0.0.1"), "", false, core.KindForbidden, true},
		{"mapped ipv4 matches", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("::ffff:192.168.1.7"), "admin", true, 0, false},
		{"bare address entry", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("::1"), "admin", true, 0, false},
		{"auth on, user", Config{UsersFile: users, Wyoming: WyomingConfig{User: "ha"}}, tcp("10.0.0.1"), "ha", false, 0, false},
		{"auth on, no user configured", Config{UsersFile: users}, tcp("10.0.0.1"), "", false, core.KindUnauthenticated, true},
		{"auth on, unknown user", Config{UsersFile: users, Wyoming: WyomingConfig{User: "nobody"}}, tcp("10.0.0.1"), "", false, core.KindUnauthenticated, true},
		{"auth on, disabled user", Config{UsersFile: users, Wyoming: WyomingConfig{User: "carol"}}, tcp("10.0.0.1"), "", false, core.KindUnauthenticated, true},
		{"nil remote with allowlist", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, nil, "", false, core.KindForbidden, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.cfg.PollInterval = -1
			a := newAuth(t, tc.cfg)
			var resolve func(net.Addr) (*core.Identity, error) = a.WyomingIdentity
			id, err := resolve(tc.remote)
			if tc.wantErr {
				if err == nil || core.KindOf(err) != tc.wantKind {
					t.Fatalf("err = %v, want kind %v", err, tc.wantKind)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if id.User != tc.wantUser || id.NetworkAdmin != tc.wantAdmin {
				t.Errorf("identity = %+v", id)
			}
		})
	}
}

func TestWyomingUserModelsApply(t *testing.T) {
	a := newAuth(t, Config{UsersFile: writeFile(t, validUsers(t)), PollInterval: -1, Wyoming: WyomingConfig{User: "ha"}})
	id, err := a.WyomingIdentity(tcp("10.0.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	if !a.AllowModel(id, "large-v3", false) {
		t.Error("ha has models [*] and should get large-v3")
	}
}

func TestBadCIDRIsAConfigError(t *testing.T) {
	if _, err := New(Config{Wyoming: WyomingConfig{AllowCIDRs: []string{"not-a-net"}}}, nil); err == nil {
		t.Error("expected an error for a bad CIDR")
	}
}
