package auth

import (
	"log/slog"
	"net"
	"net/netip"

	"github.com/GRbit/whisper-hub/internal/core"
)

// WyomingIdentity resolves the identity a Wyoming connection from remote
// acts as. Wyoming carries no credentials, so the source allowlist and the
// configured user are its only access control. Errors are *core.Error:
// KindForbidden for a source outside wyoming.allow_cidrs, KindUnauthenticated
// when no usable user is configured.
func (a *Auth) WyomingIdentity(remote net.Addr) (*core.Identity, error) {
	addr, ok := addrOf(remote)
	if len(a.wyoCIDRs) > 0 {
		allowed := false
		if ok {
			for _, p := range a.wyoCIDRs {
				if p.Contains(addr) {
					allowed = true
					break
				}
			}
		}
		if !allowed {
			slog.Warn("auth: wyoming connection refused, source not in allow_cidrs", "remote", addrString(remote))
			return nil, core.Errorf(core.KindForbidden, "source address not allowed")
		}
	}

	if a.authn == nil {
		name := a.cfg.Wyoming.User
		if name == "" {
			name = "anonymous"
		}
		slog.Debug("auth: wyoming connection accepted as anonymous", "remote", addrString(remote), "user", name)
		return &core.Identity{User: name, Anonymous: true}, nil
	}

	name := a.cfg.Wyoming.User
	if name == "" {
		slog.Warn("auth: wyoming connection refused, wyoming.user is not configured", "remote", addrString(remote))
		return nil, core.Errorf(core.KindUnauthenticated, "wyoming is not mapped to a user")
	}
	u, known := a.authn.store.get().lookup(name)
	if !known || u.Disabled {
		slog.Warn("auth: wyoming connection refused, configured user is missing or disabled",
			"remote", addrString(remote), "user", name)
		return nil, core.Errorf(core.KindUnauthenticated, "wyoming user is not available")
	}
	slog.Debug("auth: wyoming connection accepted", "remote", addrString(remote), "user", name)
	return identityOf(u), nil
}

func addrString(a net.Addr) string {
	if a == nil {
		return ""
	}
	return a.String()
}

// addrOf unmaps IPv4-in-IPv6 so "::ffff:192.168.1.5" matches 192.168.1.0/24.
func addrOf(a net.Addr) (netip.Addr, bool) {
	switch v := a.(type) {
	case nil:
		return netip.Addr{}, false
	case *net.TCPAddr:
		ip, ok := netip.AddrFromSlice(v.IP)
		return ip.Unmap(), ok
	case *net.UDPAddr:
		ip, ok := netip.AddrFromSlice(v.IP)
		return ip.Unmap(), ok
	}
	if ap, err := netip.ParseAddrPort(a.String()); err == nil {
		return ap.Addr().Unmap(), true
	}
	if ip, err := netip.ParseAddr(a.String()); err == nil {
		return ip.Unmap(), true
	}
	return netip.Addr{}, false
}
