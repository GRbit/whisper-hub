package wyoming

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strings"
	"sync"
	"time"
)

// A minimal mDNS responder (RFC 6762/6763) for one _wyoming._tcp service,
// which is all Home Assistant's zeroconf discovery needs. Hand-written to
// keep the module free of dependencies; x/net would have forced go 1.26.

const (
	mdnsPort       = 5353
	wyomingService = "_wyoming._tcp.local."
	servicesEnum   = "_services._dns-sd._udp.local."

	dnsTypeA   = 1
	dnsTypePTR = 12
	dnsTypeTXT = 16
	dnsTypeSRV = 33
	dnsTypeANY = 255
	dnsClassIN = 1
	// cacheFlush marks records only we own (RFC 6762 section 10.2).
	cacheFlush = 0x8000

	ttlHost  = 120
	ttlOther = 4500
)

var mdnsGroup = net.IPv4(224, 0, 0, 251)

type mdnsRecord struct {
	name   string
	typ    uint16
	unique bool
	ttl    uint32
	rdata  []byte
}

type responder struct {
	log      *slog.Logger
	instance string // "<name>._wyoming._tcp.local."
	host     string // "<label>.local."
	ptr      mdnsRecord
	enum     mdnsRecord
	srv      mdnsRecord
	txt      mdnsRecord
	a        mdnsRecord
}

func newResponder(name string, ip net.IP, port int, log *slog.Logger) *responder {
	r := &responder{
		log:      log,
		instance: escapeLabel(name) + "." + wyomingService,
		host:     hostLabel(name) + ".local.",
	}
	r.ptr = mdnsRecord{name: wyomingService, typ: dnsTypePTR, ttl: ttlOther, rdata: encodeName(nil, r.instance)}
	r.enum = mdnsRecord{name: servicesEnum, typ: dnsTypePTR, ttl: ttlOther, rdata: encodeName(nil, wyomingService)}
	srv := binary.BigEndian.AppendUint16(nil, 0) // priority
	srv = binary.BigEndian.AppendUint16(srv, 0)  // weight
	srv = binary.BigEndian.AppendUint16(srv, uint16(port))
	r.srv = mdnsRecord{name: r.instance, typ: dnsTypeSRV, unique: true, ttl: ttlHost, rdata: encodeName(srv, r.host)}
	// DNS-SD requires a TXT record even when empty: one zero-length string.
	r.txt = mdnsRecord{name: r.instance, typ: dnsTypeTXT, unique: true, ttl: ttlOther, rdata: []byte{0}}
	r.a = mdnsRecord{name: r.host, typ: dnsTypeA, unique: true, ttl: ttlHost, rdata: ip.To4()}
	return r
}

// announcement is the unsolicited response sent at startup, or the goodbye
// (all TTLs zero) sent at shutdown.
func (r *responder) announcement(goodbye bool) []byte {
	recs := []mdnsRecord{r.ptr, r.srv, r.txt, r.a}
	if goodbye {
		for i := range recs {
			recs[i].ttl = 0
		}
	}
	return buildResponse(0, nil, recs, nil, true)
}

// answer returns the reply to a query packet, or nil when it asks nothing
// about us. legacy queries (source port not 5353) get a unicast-style reply
// that echoes the id and questions, as RFC 6762 section 6.7 requires.
func (r *responder) answer(pkt []byte, legacy bool) []byte {
	id, qs, err := parseQuery(pkt)
	if err != nil || len(qs) == 0 {
		return nil
	}
	var answers, extra []mdnsRecord
	added := map[*mdnsRecord]bool{}
	add := func(list *[]mdnsRecord, rec *mdnsRecord) {
		if !added[rec] {
			added[rec] = true
			*list = append(*list, *rec)
		}
	}
	matches := func(q question, rec *mdnsRecord) bool {
		return strings.EqualFold(q.name, rec.name) && (q.typ == rec.typ || q.typ == dnsTypeANY)
	}
	var asked []question
	for _, q := range qs {
		hit := false
		for _, rec := range []*mdnsRecord{&r.ptr, &r.enum, &r.srv, &r.txt, &r.a} {
			if matches(q, rec) {
				add(&answers, rec)
				hit = true
			}
		}
		if hit {
			asked = append(asked, q)
		}
	}
	if len(answers) == 0 {
		return nil
	}
	// Additional records save the browser a round trip per step (RFC 6763 12).
	if added[&r.ptr] {
		add(&extra, &r.srv)
		add(&extra, &r.txt)
	}
	if added[&r.srv] {
		add(&extra, &r.a)
	}
	if !legacy {
		id, asked = 0, nil
	}
	return buildResponse(id, asked, answers, extra, !legacy)
}

type question struct {
	name string
	typ  uint16
}

func parseQuery(pkt []byte) (uint16, []question, error) {
	if len(pkt) < 12 {
		return 0, nil, errors.New("short packet")
	}
	id := binary.BigEndian.Uint16(pkt)
	flags := binary.BigEndian.Uint16(pkt[2:])
	if flags&0x8000 != 0 || flags&0x7800 != 0 { // responses and non-QUERY opcodes
		return 0, nil, errors.New("not a query")
	}
	qd := int(binary.BigEndian.Uint16(pkt[4:]))
	off := 12
	qs := make([]question, 0, min(qd, 16))
	for range qd {
		name, next, err := decodeName(pkt, off)
		if err != nil {
			return 0, nil, err
		}
		if next+4 > len(pkt) {
			return 0, nil, errors.New("truncated question")
		}
		qs = append(qs, question{name: name, typ: binary.BigEndian.Uint16(pkt[next:])})
		off = next + 4
	}
	return id, qs, nil
}

// decodeName reads a possibly compressed name at off and returns it with the
// offset just past it in the original stream.
func decodeName(pkt []byte, off int) (string, int, error) {
	var labels []string
	end := -1
	for jumps := 0; ; {
		if off >= len(pkt) {
			return "", 0, errors.New("name out of range")
		}
		l := int(pkt[off])
		switch {
		case l == 0:
			if end < 0 {
				end = off + 1
			}
			return strings.Join(labels, ".") + ".", end, nil
		case l&0xC0 == 0xC0:
			if off+1 >= len(pkt) {
				return "", 0, errors.New("truncated pointer")
			}
			if end < 0 {
				end = off + 2
			}
			if jumps++; jumps > 16 {
				return "", 0, errors.New("compression loop")
			}
			off = int(binary.BigEndian.Uint16(pkt[off:]) & 0x3FFF)
		case l&0xC0 != 0:
			return "", 0, errors.New("bad label type")
		default:
			if off+1+l > len(pkt) {
				return "", 0, errors.New("truncated label")
			}
			labels = append(labels, string(pkt[off+1:off+1+l]))
			off += 1 + l
		}
	}
}

func buildResponse(id uint16, qs []question, answers, extra []mdnsRecord, multicast bool) []byte {
	b := binary.BigEndian.AppendUint16(nil, id)
	b = binary.BigEndian.AppendUint16(b, 0x8400) // response, authoritative
	b = binary.BigEndian.AppendUint16(b, uint16(len(qs)))
	b = binary.BigEndian.AppendUint16(b, uint16(len(answers)))
	b = binary.BigEndian.AppendUint16(b, 0)
	b = binary.BigEndian.AppendUint16(b, uint16(len(extra)))
	for _, q := range qs {
		b = encodeName(b, q.name)
		b = binary.BigEndian.AppendUint16(b, q.typ)
		b = binary.BigEndian.AppendUint16(b, dnsClassIN)
	}
	for _, rec := range append(answers, extra...) {
		class := uint16(dnsClassIN)
		ttl := rec.ttl
		if multicast && rec.unique {
			class |= cacheFlush
		}
		if !multicast {
			ttl = min(ttl, 10) // legacy unicast replies, RFC 6762 6.7
		}
		b = encodeName(b, rec.name)
		b = binary.BigEndian.AppendUint16(b, rec.typ)
		b = binary.BigEndian.AppendUint16(b, class)
		b = binary.BigEndian.AppendUint32(b, ttl)
		b = binary.BigEndian.AppendUint16(b, uint16(len(rec.rdata)))
		b = append(b, rec.rdata...)
	}
	return b
}

// encodeName writes name uncompressed. Labels were built by escapeLabel or
// are constants, so a "." inside a label cannot occur.
func encodeName(b []byte, name string) []byte {
	for _, l := range strings.Split(strings.TrimSuffix(name, "."), ".") {
		if l == "" {
			continue
		}
		b = append(b, byte(len(l)))
		b = append(b, l...)
	}
	return append(b, 0)
}

// escapeLabel makes an instance label: any UTF-8 is allowed there, but dots
// would split it and DNS caps labels at 63 bytes.
func escapeLabel(name string) string {
	name = strings.ReplaceAll(name, ".", "-")
	if len(name) > 63 {
		name = name[:63]
	}
	if name == "" {
		name = defaultProgramName
	}
	return name
}

func hostLabel(name string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(name) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return escapeLabel(strings.Trim(b.String(), "-"))
}

// detectIP picks the address the kernel would use to reach the mDNS group,
// the same trick wyoming's zeroconf.py uses.
func detectIP() (net.IP, error) {
	c, err := net.DialUDP("udp4", nil, &net.UDPAddr{IP: mdnsGroup, Port: 1})
	if err != nil {
		return nil, err
	}
	defer c.Close()
	return c.LocalAddr().(*net.UDPAddr).IP.To4(), nil
}

func (s *Server) startZeroconf(ctx context.Context, addr net.Addr) (func(), error) {
	tcp, ok := addr.(*net.TCPAddr)
	if !ok {
		return nil, fmt.Errorf("listener is not TCP: %T", addr)
	}
	var ip net.IP
	if h := s.cfg.Zeroconf.Host; h != "" {
		if ip = net.ParseIP(h).To4(); ip == nil {
			return nil, fmt.Errorf("zeroconf host %q is not an IPv4 address", h)
		}
	} else {
		var err error
		if ip, err = detectIP(); err != nil {
			return nil, fmt.Errorf("detect address: %w", err)
		}
	}
	pc, err := listenMDNS(ip)
	if err != nil {
		return nil, err
	}
	r := newResponder(s.cfg.Zeroconf.Name, ip, tcp.Port, s.log)
	group := &net.UDPAddr{IP: mdnsGroup, Port: mdnsPort}

	ctx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		// RFC 6762 8.3: announce at least twice, one second apart.
		for i := range 2 {
			if _, err := pc.WriteToUDP(r.announcement(false), group); err != nil {
				s.log.Warn("wyoming zeroconf announce failed", "err", err)
			}
			if i == 0 {
				select {
				case <-time.After(time.Second):
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	go func() {
		defer wg.Done()
		buf := make([]byte, 9000)
		for {
			n, src, err := pc.ReadFromUDP(buf)
			if err != nil {
				if ctx.Err() == nil {
					s.log.Warn("wyoming zeroconf read failed", "err", err)
				}
				return
			}
			legacy := src.Port != mdnsPort
			resp := r.answer(buf[:n], legacy)
			if resp == nil {
				continue
			}
			dst := group
			if legacy {
				dst = src
			}
			s.log.Debug("wyoming zeroconf query answered", "from", src.String(), "legacy", legacy)
			if _, err := pc.WriteToUDP(resp, dst); err != nil {
				s.log.Debug("wyoming zeroconf reply failed", "err", err)
			}
		}
	}()
	s.log.Info("wyoming zeroconf announced", "instance", r.instance, "host", r.host, "ip", ip.String(), "port", tcp.Port)

	return func() {
		cancel()
		_, _ = pc.WriteToUDP(r.announcement(true), group)
		_ = pc.Close()
		wg.Wait()
		s.log.Debug("wyoming zeroconf goodbye sent")
	}, nil
}
