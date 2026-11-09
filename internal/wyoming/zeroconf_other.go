//go:build !linux

package wyoming

import (
	"errors"
	"net"
)

// The gateway ships as a Linux container; elsewhere zeroconf reports itself
// unavailable and Serve carries on without it.
func listenMDNS(net.IP) (*net.UDPConn, error) {
	return nil, errors.New("zeroconf is only supported on linux")
}
