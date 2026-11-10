package auth

import (
	"log/slog"
	"net/http"
	"path"
	"strconv"
	"strings"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Middleware authenticates requests to protected paths and stores the
// caller in the request context (read it with core.IdentityFrom). Public
// paths, OPTIONS requests (CORS preflight), and every path while
// authentication is off, pass through with admin access
// (core.Identity.NetworkAdmin).
//
// Failures are written by the rejection writer registered for the path
// (see New), so each API answers in its own dialect; unregistered paths get
// a plain-text 401 with a Basic challenge. The
// error passed to a writer is a *core.Error of KindUnauthenticated with a
// client-safe message; Retry-After is already set when the client is rate
// limited.
func (a *Auth) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// CORS preflights never carry credentials; the handlers behind
		// (method patterns) run no job for OPTIONS.
		if a.authn == nil || r.Method == http.MethodOptions || !a.isProtected(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		addr := clientAddr(r, a.cfg.TrustForwarded)
		res := a.authn.authenticate(r, addr)
		if res.outcome != outcomeOK {
			a.logRejection(r, addr, res)
			a.rec.recordAuthFailure(res.attempted, protocolOf(r.URL.Path), r.URL.Path, addr, res.outcome.String())
			a.reject(w, r, res.outcome)
			return
		}
		slog.Debug("auth: authenticated", "user", res.user.Name, "credential", res.cred,
			"cached", res.cached, "method", r.Method, "path", r.URL.Path, "client_ip", addr)
		next.ServeHTTP(w, r.WithContext(core.WithIdentity(r.Context(), identityOf(res.user))))
	})
}

func (a *Auth) logRejection(r *http.Request, addr string, res authResult) {
	attrs := []any{"credential", res.cred, "reason", res.outcome.String(),
		"attempted_user", res.attempted, "method", r.Method, "path", r.URL.Path, "client_ip", addr}
	switch {
	case res.blockedNow:
		slog.Warn("auth: too many failed attempts, blocking address",
			append(attrs, "window", a.cfg.RateLimit.Window)...)
	case res.outcome == outcomeLimited:
		// Already announced when the block started; Warn here would let an
		// attacker flood the log.
		slog.Debug("auth: rejected, address is rate limited", attrs...)
	case res.outcome == outcomeMissing:
		slog.Debug("auth: rejected, no credentials", attrs...)
	default:
		slog.Warn("auth: authentication rejected", attrs...)
	}
}

// isProtected matches on the cleaned path so "/v1/../asr" cannot dodge the
// policy; the mux would serve it as "/asr" after its own redirect.
func (a *Auth) isProtected(p string) bool {
	p = path.Clean("/" + p)
	for _, e := range a.protected {
		if matchPrefix(e, p) {
			return true
		}
	}
	return false
}

// matchPrefix reports whether cleaned path p is entry or below it; "/v1/"
// also matches "/v1".
func matchPrefix(entry, p string) bool {
	base := strings.TrimSuffix(entry, "/")
	return p == base || strings.HasPrefix(p, base+"/")
}

func isOpenAIPath(p string) bool { return p == "/v1" || strings.HasPrefix(p, "/v1/") }

func protocolOf(p string) string {
	if isOpenAIPath(p) {
		return core.ProtocolOpenAI
	}
	return core.ProtocolLegacy
}

// Client-safe messages handed to the rejection writer. The default writer
// ignores them and sends a fixed body.
const (
	msgMissing = "No credentials provided. Send an API key as \"Authorization: Bearer <key>\" or use HTTP Basic."
	msgInvalid = "Incorrect API key or credentials provided."
	msgLimited = "Too many failed authentication attempts, retry later."
)

func (a *Auth) reject(w http.ResponseWriter, r *http.Request, o outcome) {
	msg := msgInvalid
	switch o {
	case outcomeMissing:
		msg = msgMissing
	case outcomeLimited:
		msg = msgLimited
		// The block may end sooner (fixed windows), but never later.
		w.Header().Set("Retry-After", strconv.Itoa(int(a.cfg.RateLimit.Window.Seconds())))
	}
	a.rejecterFor(r.URL.Path)(w, core.Errorf(core.KindUnauthenticated, "%s", msg))
}

// rejecterFor picks the most specific configured prefix, so "/v1/" can be
// overridden for "/v1/admin/" without listing every other path.
func (a *Auth) rejecterFor(p string) func(http.ResponseWriter, error) {
	p = path.Clean("/" + p)
	best, bestLen := a.challenge, -1
	for prefix, fn := range a.rejecters {
		if matchPrefix(prefix, p) && len(prefix) > bestLen {
			best, bestLen = fn, len(prefix)
		}
	}
	return best
}

// challenge writes the default 401. The header is what makes a browser
// prompt for credentials rather than show an error.
func (a *Auth) challenge(w http.ResponseWriter, _ error) {
	w.Header().Set("WWW-Authenticate", `Basic realm="`+a.cfg.Realm+`", charset="UTF-8"`)
	http.Error(w, "unauthorized", http.StatusUnauthorized)
}
