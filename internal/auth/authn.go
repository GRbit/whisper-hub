package auth

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"hash"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/bcrypt"
)

// Only matters when the file is edited without a reload being noticed; a
// reload drops the cache outright.
const credCacheTTL = 5 * time.Minute

// Valid entries are bounded by the number of real logins; the cap only
// guards against a bug turning the cache into a leak.
const credCacheMax = 4096

// credCache remembers verified Basic credentials, because bcrypt at cost 12
// costs ~250ms by design, which is right against offline cracking and wrong
// on every upload.
//
// Keys are HMAC-SHA256 under a per-process random secret over the
// length-prefixed name, password and stored hash: no two distinct inputs
// share a key, a password change invalidates the entry even before a
// reload, and a memory dump does not yield an unsalted fast hash of the
// password.
type credCache struct {
	secret [32]byte

	mu      sync.Mutex
	entries map[[sha256.Size]byte]time.Time
	now     func() time.Time // injectable for tests
}

func newCredCache() *credCache {
	c := &credCache{entries: make(map[[sha256.Size]byte]time.Time), now: time.Now}
	if _, err := rand.Read(c.secret[:]); err != nil {
		// crypto/rand does not fail on supported platforms.
		panic("auth: crypto/rand: " + err.Error())
	}
	return c
}

func writeField(h hash.Hash, s string) {
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], uint64(len(s)))
	h.Write(n[:])
	h.Write([]byte(s))
}

func (c *credCache) key(name, password, storedHash string) [sha256.Size]byte {
	m := hmac.New(sha256.New, c.secret[:])
	writeField(m, name)
	writeField(m, password)
	writeField(m, storedHash)
	var out [sha256.Size]byte
	m.Sum(out[:0])
	return out
}

func (c *credCache) valid(name, password, storedHash string) bool {
	k := c.key(name, password, storedHash)
	c.mu.Lock()
	defer c.mu.Unlock()
	expiry, ok := c.entries[k]
	if !ok {
		return false
	}
	if c.now().After(expiry) {
		delete(c.entries, k)
		return false
	}
	return true
}

func (c *credCache) remember(name, password, storedHash string) {
	k := c.key(name, password, storedHash)
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	if len(c.entries) >= credCacheMax {
		for ek, exp := range c.entries {
			if now.After(exp) {
				delete(c.entries, ek)
			}
		}
		if len(c.entries) >= credCacheMax {
			clear(c.entries)
		}
	}
	c.entries[k] = now.Add(credCacheTTL)
}

// Called on reload, so a deleted user stops being accepted at once rather
// than after the TTL.
func (c *credCache) clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	clear(c.entries)
}

// Tracking stops for new addresses past this many; they then just cost
// bcrypt time like they would without a limiter.
const limiterMaxAddrs = 65536

type failWindow struct {
	start time.Time
	n     int
}

// limiter counts failed attempts per client address in fixed windows.
// Fixed windows are coarse but need one map entry per address and no timers.
type limiter struct {
	max    int
	window time.Duration
	now    func() time.Time

	mu        sync.Mutex
	addrs     map[string]*failWindow
	lastPrune time.Time
}

func newLimiter(cfg RateLimitConfig) *limiter {
	if cfg.Failures < 0 {
		return nil
	}
	return &limiter{max: cfg.Failures, window: cfg.Window, now: time.Now, addrs: make(map[string]*failWindow)}
}

func (l *limiter) blocked(addr string) bool {
	if l == nil {
		return false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	w := l.addrs[addr]
	if w == nil {
		return false
	}
	if l.now().Sub(w.start) >= l.window {
		delete(l.addrs, addr)
		return false
	}
	return w.n >= l.max
}

// fail records a failure and reports whether it was the one that blocked
// addr, so the block is logged once rather than per request.
func (l *limiter) fail(addr string) bool {
	if l == nil {
		return false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	w := l.addrs[addr]
	if w != nil && now.Sub(w.start) >= l.window {
		w = nil
	}
	if w == nil {
		if len(l.addrs) >= limiterMaxAddrs && !l.prune(now) {
			return false
		}
		w = &failWindow{start: now}
		l.addrs[addr] = w
	}
	w.n++
	return w.n == l.max
}

// Rate limited to once a second so a flood of distinct addresses cannot make
// every failure an O(n) scan.
func (l *limiter) prune(now time.Time) bool {
	if now.Sub(l.lastPrune) < time.Second {
		return false
	}
	l.lastPrune = now
	for a, w := range l.addrs {
		if now.Sub(w.start) >= l.window {
			delete(l.addrs, a)
		}
	}
	return len(l.addrs) < limiterMaxAddrs
}

func (l *limiter) reset(addr string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.addrs, addr)
}

type outcome int

const (
	outcomeOK      outcome = iota
	outcomeMissing         // no credentials at all
	outcomeInvalid         // wrong, unknown, disabled or malformed
	outcomeLimited         // too many failures from this address
)

func (o outcome) String() string {
	switch o {
	case outcomeOK:
		return "ok"
	case outcomeMissing:
		return "missing"
	case outcomeLimited:
		return "rate_limited"
	default:
		return "invalid"
	}
}

// Credential kinds, as they appear in logs.
const (
	credBasic   = "basic"
	credBearer  = "bearer"
	credXAPIKey = "x-api-key"
	credOther   = "unsupported"
)

type authResult struct {
	user    *account
	outcome outcome
	// cred is the kind of credential presented; never the credential.
	cred string
	// attempted is the claimed Basic login, kept on failure so brute force
	// is visible in the logs. Never set for API keys.
	attempted string
	cached    bool
	// blockedNow is set on the failure that tripped the rate limit, so the
	// block is logged once.
	blockedNow bool
}

// authenticator verifies request credentials against the current users.
type authenticator struct {
	store   *store
	cache   *credCache
	limiter *limiter
}

func (a *authenticator) authenticate(r *http.Request, addr string) authResult {
	if h := r.Header.Get("Authorization"); h != "" {
		scheme, rest, _ := strings.Cut(h, " ")
		switch {
		case strings.EqualFold(scheme, "Basic"):
			name, password, ok := r.BasicAuth()
			if !ok {
				return a.fail(authResult{cred: credBasic}, addr)
			}
			return a.checkBasic(name, password, addr)
		case strings.EqualFold(scheme, "Bearer"):
			return a.checkKey(strings.TrimSpace(rest), credBearer, addr)
		default:
			return a.fail(authResult{cred: credOther}, addr)
		}
	}
	if k := r.Header.Get("X-Api-Key"); k != "" {
		return a.checkKey(strings.TrimSpace(k), credXAPIKey, addr)
	}
	return authResult{outcome: outcomeMissing}
}

func (a *authenticator) fail(res authResult, addr string) authResult {
	res.user = nil
	if a.limiter.blocked(addr) {
		res.outcome = outcomeLimited
		return res
	}
	res.outcome = outcomeInvalid
	res.blockedNow = a.limiter.fail(addr)
	return res
}

func (a *authenticator) checkKey(key, cred, addr string) authResult {
	u := a.store.get()
	sum := sha256.Sum256([]byte(key))
	var match *account
	// Every key is compared, with no early exit, so timing says nothing
	// about which key or how many keys exist.
	for i := range u.keys {
		if subtle.ConstantTimeCompare(sum[:], u.keys[i].sum[:]) == 1 {
			match = u.keys[i].user
		}
	}
	if key == "" || match == nil || match.Disabled {
		return a.fail(authResult{cred: cred}, addr)
	}
	return authResult{user: match, outcome: outcomeOK, cred: cred}
}

func (a *authenticator) checkBasic(name, password, addr string) authResult {
	u := a.store.get()
	usr, known := u.lookup(name)
	hasHash := known && usr.PasswordHash != ""

	// A cache hit is unforgeable (see credCache) and costs no bcrypt time,
	// so it is honoured even from a rate limited address.
	if hasHash && !usr.Disabled && a.cache.valid(name, password, usr.PasswordHash) {
		return authResult{user: usr, outcome: outcomeOK, cred: credBasic, cached: true}
	}
	if a.limiter.blocked(addr) {
		return authResult{outcome: outcomeLimited, cred: credBasic, attempted: name}
	}

	hash := u.dummyHash()
	if hasHash {
		hash = []byte(usr.PasswordHash)
	}
	// Run even for unknown users to keep timing flat.
	err := bcrypt.CompareHashAndPassword(hash, []byte(password))
	if !hasHash || err != nil || usr.Disabled {
		return authResult{outcome: outcomeInvalid, cred: credBasic, attempted: name, blockedNow: a.limiter.fail(addr)}
	}
	a.cache.remember(name, password, usr.PasswordHash)
	a.limiter.reset(addr)
	return authResult{user: usr, outcome: outcomeOK, cred: credBasic}
}

// clientAddr returns the peer address, preferring a forwarded header only
// when the operator said an upstream proxy can be trusted. Honouring it
// unconditionally would let any client pick its own rate limit bucket.
func clientAddr(r *http.Request, trustForwarded bool) string {
	if trustForwarded {
		if fwd := r.Header.Get("X-Forwarded-For"); fwd != "" {
			// Leftmost entry is the original client.
			first, _, _ := strings.Cut(fwd, ",")
			return strings.TrimSpace(first)
		}
		if real := r.Header.Get("X-Real-IP"); real != "" {
			return strings.TrimSpace(real)
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
