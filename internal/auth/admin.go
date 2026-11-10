package auth

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/GRbit/whisper-hub/internal/core"
)

// UsagePath is the suggested mount point for UsageHandler.
const UsagePath = "/_hub/usage"

type usageResponse struct {
	Users map[string]UserTotals `json:"users"`
}

// UsageHandler serves {"users": {name: UserTotals}} as JSON to admin access
// by network (every request while authentication is off, and requests on
// the AdminPort listener) and to users flagged admin; "?user=<name>"
// narrows it to one user. On the authenticated listener it authenticates by
// itself, whatever the path policy says, and is subject to the same rate
// limit.
func (a *Auth) UsageHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		user := core.NetworkAdminUser
		if adminPort, _ := r.Context().Value(adminPortKey{}).(bool); a.authn != nil && !adminPort {
			addr := clientAddr(r, a.cfg.TrustForwarded)
			res := a.authn.authenticate(r, addr)
			if res.outcome != outcomeOK {
				a.logRejection(r, addr, res)
				a.reject(w, r, res.outcome)
				return
			}
			if !res.user.Admin {
				slog.Warn("auth: usage request from a non-admin user", "user", res.user.Name, "client_ip", addr)
				http.Error(w, "admin only", http.StatusForbidden)
				return
			}
			user = res.user.Name
		}

		totals := a.rec.Totals()
		if only := r.URL.Query().Get("user"); only != "" {
			t, ok := totals[only]
			totals = map[string]UserTotals{}
			if ok {
				totals[only] = t
			}
		}
		slog.Debug("auth: usage served", "user", user, "entries", len(totals))
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		_ = enc.Encode(usageResponse{Users: totals})
	})
}

type adminPortKey struct{}

// AdminPort wraps the handler of a listener without authentication, the
// admin port (http.open_listen): its requests have admin access, which
// UsageHandler honours. It deliberately skips the middleware.
func (a *Auth) AdminPort(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), adminPortKey{}, true)))
	})
}
