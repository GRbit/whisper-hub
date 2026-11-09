package wyoming

import (
	"encoding/binary"
	"net"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestZeroconfAnswersQuery runs Serve with zeroconf on and asks for the
// service the way a legacy (non-5353) querier does: one multicast query,
// answered by unicast. It needs a multicast-capable interface, so it skips
// on hosts without one rather than failing.
func TestZeroconfAnswersQuery(t *testing.T) {
	probe, err := net.DialUDP("udp4", nil, &net.UDPAddr{IP: mdnsGroup, Port: 1})
	if err != nil {
		t.Skipf("no route to the mDNS group: %v", err)
	}
	ip := probe.LocalAddr().(*net.UDPAddr).IP.To4()
	_ = probe.Close()

	h := startServer(t, Config{Zeroconf: ZeroconfConfig{Enabled: true, Name: "Test Hub.1"}}, newFakeService(), nil)
	_, portStr, _ := net.SplitHostPort(h.addr)

	q, err := net.ListenUDP("udp4", &net.UDPAddr{IP: ip})
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	query := binary.BigEndian.AppendUint16(nil, 0x1234)
	query = append(query, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
	query = encodeName(query, wyomingService)
	query = append(query, 0, dnsTypePTR, 0, dnsClassIN)

	buf := make([]byte, 9000)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := q.WriteToUDP(query, &net.UDPAddr{IP: mdnsGroup, Port: mdnsPort}); err != nil {
			t.Fatal(err)
		}
		_ = q.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
		n, _, err := q.ReadFromUDP(buf)
		if err != nil {
			continue
		}
		resp := string(buf[:n])
		if binary.BigEndian.Uint16(buf) != 0x1234 {
			continue // another responder on the host
		}
		port := make([]byte, 2)
		binary.BigEndian.PutUint16(port, uint16(port16(t, portStr)))
		for name, want := range map[string]string{
			"instance": "\x0aTest Hub-1\x08_wyoming\x04_tcp\x05local\x00",
			"host":     "\x0atest-hub-1\x05local\x00",
			"port":     string(port),
			"address":  string(ip),
		} {
			if !strings.Contains(resp, want) {
				t.Errorf("response lacks %s %q", name, want)
			}
		}
		return
	}
	t.Fatal("no mDNS answer from the server")
}

func port16(t *testing.T, s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatal(err)
	}
	return n
}
