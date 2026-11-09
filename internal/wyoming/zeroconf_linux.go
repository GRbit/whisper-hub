package wyoming

import (
	"context"
	"fmt"
	"net"
	"syscall"
)

// listenMDNS binds 0.0.0.0:5353 next to any other mDNS stack on the host
// (avahi, browsers) and joins the group on the interface owning ip. Replies
// must leave from port 5353 (RFC 6762 11), so the same socket sends.
func listenMDNS(ip net.IP) (*net.UDPConn, error) {
	ifaceIP := [4]byte(ip.To4())
	lc := net.ListenConfig{Control: func(_, _ string, rc syscall.RawConn) error {
		var serr error
		err := rc.Control(func(fd uintptr) {
			s := int(fd)
			for _, o := range [][3]int{
				{syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1},
				{syscall.IPPROTO_IP, syscall.IP_MULTICAST_TTL, 255},
				{syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, 1},
			} {
				if serr = syscall.SetsockoptInt(s, o[0], o[1], o[2]); serr != nil {
					return
				}
			}
			mreq := &syscall.IPMreq{Multiaddr: [4]byte(mdnsGroup.To4()), Interface: ifaceIP}
			if serr = syscall.SetsockoptIPMreq(s, syscall.IPPROTO_IP, syscall.IP_ADD_MEMBERSHIP, mreq); serr != nil {
				serr = fmt.Errorf("join mDNS group: %w", serr)
				return
			}
			serr = syscall.SetsockoptInet4Addr(s, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, ifaceIP)
		})
		if err != nil {
			return err
		}
		return serr
	}}
	pc, err := lc.ListenPacket(context.Background(), "udp4", fmt.Sprintf("0.0.0.0:%d", mdnsPort))
	if err != nil {
		return nil, fmt.Errorf("mDNS socket: %w", err)
	}
	return pc.(*net.UDPConn), nil
}
