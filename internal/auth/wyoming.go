package auth

import (
	"log/slog"
	"net"
	"net/netip"

	"github.com/GRbit/whisper-hub/internal/core"
)

// WyomingIdentity resolves the identity a Wyoming connection from remote
// acts as: always core.WyomingUser. Wyoming carries no credentials, so the
// network (and wyoming.allow_cidrs) decides who connects; a users file
// entry named core.WyomingUser applies its models patterns and disabled
// flag. Errors are *core.Error: KindForbidden for a source outside
// wyoming.allow_cidrs, KindUnauthenticated when that entry is disabled.
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

	// Unrestricted unless a users file entry named WyomingUser says
	// otherwise: restrictions are for users, and the network already
	// decided who may connect.
	unrestricted := &core.Identity{User: core.WyomingUser, Models: []string{"*"}}
	if a.authn == nil {
		slog.Debug("auth: wyoming connection accepted", "remote", addrString(remote), "user", core.WyomingUser)
		return unrestricted, nil
	}
	u, known := a.authn.store.get().lookup(core.WyomingUser)
	if !known {
		slog.Debug("auth: wyoming connection accepted, no users file entry restricts it",
			"remote", addrString(remote), "user", core.WyomingUser)
		return unrestricted, nil
	}
	if u.Disabled {
		slog.Warn("auth: wyoming connection refused, its users file entry is disabled",
			"remote", addrString(remote), "user", core.WyomingUser)
		return nil, core.Errorf(core.KindUnauthenticated, "wyoming is disabled")
	}
	slog.Debug("auth: wyoming connection accepted", "remote", addrString(remote), "user", core.WyomingUser)
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
