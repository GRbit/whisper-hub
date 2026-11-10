package auth

import (
	"net"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

func tcp(ip string) net.Addr { return &net.TCPAddr{IP: net.ParseIP(ip), Port: 40000} }

// Wyoming has no credentials: every connection the network lets through
// acts as core.WyomingUser, and a users file entry of that name, if any,
// restricts it.
func TestWyomingIdentity(t *testing.T) {
	plain := writeFile(t, validUsers(t))
	restricted := writeFile(t, validUsers(t)+"  - name: wyoming\n    models: [\"base\"]\n")
	disabled := writeFile(t, validUsers(t)+"  - name: wyoming\n    disabled: true\n")
	cidrs := []string{"192.168.1.0/24", "::1"}

	cases := []struct {
		name     string
		cfg      Config
		remote   net.Addr
		allowed  []string // models a non-default request may use
		denied   []string
		wantKind core.ErrorKind
		wantErr  bool
	}{
		{"auth off", Config{}, tcp("10.0.0.1"), []string{"large-v3", "base"}, nil, 0, false},
		{"auth off, cidr allows", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("192.168.1.7"), []string{"large-v3"}, nil, 0, false},
		{"auth off, cidr denies", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("10.0.0.1"), nil, nil, core.KindForbidden, true},
		{"mapped ipv4 matches", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("::ffff:192.168.1.7"), []string{"large-v3"}, nil, 0, false},
		{"bare address entry", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, tcp("::1"), []string{"large-v3"}, nil, 0, false},
		{"auth on, no wyoming entry", Config{UsersFile: plain}, tcp("10.0.0.1"), []string{"large-v3"}, nil, 0, false},
		{"auth on, wyoming entry restricts", Config{UsersFile: restricted}, tcp("10.0.0.1"), []string{"base"}, []string{"large-v3"}, 0, false},
		{"auth on, wyoming entry disabled", Config{UsersFile: disabled}, tcp("10.0.0.1"), nil, nil, core.KindUnauthenticated, true},
		{"nil remote with allowlist", Config{Wyoming: WyomingConfig{AllowCIDRs: cidrs}}, nil, nil, nil, core.KindForbidden, true},
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
			if id.User != core.WyomingUser || id.NetworkAdmin {
				t.Errorf("identity = %+v", id)
			}
			for _, m := range tc.allowed {
				if !a.AllowModel(id, m, false) {
					t.Errorf("%s denied", m)
				}
			}
			for _, m := range tc.denied {
				if a.AllowModel(id, m, false) {
					t.Errorf("%s allowed", m)
				}
			}
			if !a.AllowModel(id, "tiny", true) {
				t.Error("default model denied")
			}
		})
	}
}

func TestBadCIDRIsAConfigError(t *testing.T) {
	if _, err := New(Config{Wyoming: WyomingConfig{AllowCIDRs: []string{"not-a-net"}}}, nil); err == nil {
		t.Error("expected an error for a bad CIDR")
	}
}
