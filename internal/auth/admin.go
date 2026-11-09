package auth

import (
	"encoding/json"
	"log/slog"
	"net/http"
)

// UsagePath is the suggested mount point for UsageHandler.
const UsagePath = "/_hub/usage"

type usageResponse struct {
	Users map[string]UserTotals `json:"users"`
}

// UsageHandler serves {"users": {name: UserTotals}} as JSON to users flagged
// admin; "?user=<name>" narrows it to one user. It authenticates by itself,
// whatever the path policy says, and is subject to the same rate limit.
//
// While authentication is off it answers 403: there is no admin to check
// for, and the totals file on disk holds the same data.
func (a *Auth) UsageHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if a.authn == nil {
			http.Error(w, "usage is served only while authentication is on", http.StatusForbidden)
			return
		}
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

		totals := a.rec.Totals()
		if only := r.URL.Query().Get("user"); only != "" {
			t, ok := totals[only]
			totals = map[string]UserTotals{}
			if ok {
				totals[only] = t
			}
		}
		slog.Debug("auth: usage served", "user", res.user.Name, "entries", len(totals))
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		_ = enc.Encode(usageResponse{Users: totals})
	})
}
