package auth

import (
	"context"
	"errors"
	"log/slog"
	"maps"
	"net/http"
	"net/netip"
	"slices"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Auth is the authentication, model access and usage accounting component.
// It implements core.Authorizer and core.UsageRecorder.
type Auth struct {
	cfg       Config
	authn     *authenticator // nil when authentication is off
	rec       *recorder
	wyoCIDRs  []netip.Prefix
	protected []string
	rejecters map[string]func(http.ResponseWriter, error)
}

var (
	_ core.Authorizer    = (*Auth)(nil)
	_ core.UsageRecorder = (*Auth)(nil)
)

// New builds an Auth from cfg. It fails only on configuration errors (bad
// CIDR, unusable usage paths); a missing or broken users file is not fatal
// and leaves the service up, rejecting every protected request.
//
// rejecters maps a path prefix (matched like Config.ProtectedPaths, most
// specific wins) to the writer of that API's authentication failure, for
// example {"/v1/": openaiapi.WriteError}. May be nil.
func New(cfg Config, rejecters map[string]func(w http.ResponseWriter, err error)) (*Auth, error) {
	cfg = cfg.Resolved()
	cidrs, err := parseCIDRs(cfg.Wyoming.AllowCIDRs)
	if err != nil {
		return nil, err
	}
	rec, err := newRecorder(cfg.Usage)
	if err != nil {
		return nil, err
	}
	a := &Auth{cfg: cfg, rec: rec, wyoCIDRs: cidrs, protected: cfg.ProtectedPaths,
		rejecters: maps.Clone(rejecters)}

	if cfg.UsersFile == "" {
		slog.Info("auth: authentication is off (no users file configured), every caller has admin access",
			"admin_restricted", cfg.Admin.Restrict)
		return a, nil
	}
	cache := newCredCache()
	a.authn = &authenticator{
		cache:   cache,
		limiter: newLimiter(cfg.RateLimit),
		store:   newStore(cfg.UsersFile, func(*users) { cache.clear() }),
	}
	slog.Info("auth: authentication is on", "users_file", cfg.UsersFile,
		"protected_paths", cfg.ProtectedPaths, "rate_limit_failures", cfg.RateLimit.Failures,
		"rate_limit_window", cfg.RateLimit.Window)
	if cfg.Wyoming.User == "" {
		slog.Warn("auth: wyoming.user is not set, Wyoming connections will be refused while authentication is on")
	} else if _, ok := a.authn.store.get().lookup(cfg.Wyoming.User); !ok {
		slog.Warn("auth: wyoming.user is not in the users file, Wyoming connections will be refused until it is added",
			"user", cfg.Wyoming.User)
	}
	return a, nil
}

// Reload re-reads the users file at once (wire it to SIGHUP) and reopens the
// request log, so an external logrotate can move it away. On error the
// previous users stay in force.
func (a *Auth) Reload() error {
	var err error
	if a.authn != nil {
		err = a.authn.store.refresh(true)
	}
	return errors.Join(err, a.rec.reopenLog())
}

// Run polls the users file for edits and snapshots usage totals until ctx
// is cancelled. Polling rather than inotify: editing the file in a running
// container should be enough, and inotify is unreliable across bind mounts.
func (a *Auth) Run(ctx context.Context) {
	go a.rec.Run(ctx)
	if a.authn == nil || a.cfg.PollInterval < 0 {
		<-ctx.Done()
		return
	}
	t := time.NewTicker(a.cfg.PollInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			_ = a.authn.store.refresh(false)
		}
	}
}

// Close writes a final totals snapshot and closes the request log.
func (a *Auth) Close() error { return a.rec.Close() }

// Record implements core.UsageRecorder. It works with authentication off
// too, billing the caller's name (empty -> core.NetworkAdminUser). The request log
// line is written synchronously: one write(2) of a few hundred bytes per job
// is cheap next to a transcription, but a stalled disk would stall callers.
func (a *Auth) Record(ctx context.Context, ev core.UsageEvent) { a.rec.Record(ctx, ev) }

// Totals returns a copy of the per-user usage totals.
func (a *Auth) Totals() map[string]UserTotals { return a.rec.Totals() }

// AllowModel implements core.Authorizer.
//
// Authenticated users always get the default model plus whatever their
// models patterns match. Admin access without credentials gets everything
// unless admin.restrict is set, in which case it is treated like a user
// whose patterns are admin.models.
func (a *Auth) AllowModel(id *core.Identity, model string, isDefault bool) bool {
	var allowed bool
	switch {
	case id == nil || id.NetworkAdmin:
		allowed = !a.cfg.Admin.Restrict || isDefault || matchAny(a.cfg.Admin.Models, model)
	default:
		allowed = isDefault || matchAny(id.Models, model)
	}
	if !allowed {
		user := core.NetworkAdminUser
		if id != nil {
			user = id.User
		}
		slog.Debug("auth: model denied", "user", user, "model", model)
	}
	return allowed
}

func matchAny(patterns []string, model string) bool {
	return slices.ContainsFunc(patterns, func(p string) bool { return matchGlob(p, model) })
}

// matchGlob supports only "*" (any run of characters, "/" included): model
// ids like "Systran/faster-whisper-small" make path.Match semantics wrong.
func matchGlob(pattern, s string) bool {
	px, sx := 0, 0
	star, mark := -1, 0
	for sx < len(s) {
		switch {
		case px < len(pattern) && pattern[px] == '*':
			star, mark = px, sx
			px++
		case px < len(pattern) && pattern[px] == s[sx]:
			px++
			sx++
		case star >= 0:
			px = star + 1
			mark++
			sx = mark
		default:
			return false
		}
	}
	for px < len(pattern) && pattern[px] == '*' {
		px++
	}
	return px == len(pattern)
}

func identityOf(u *account) *core.Identity {
	return &core.Identity{User: u.Name, Models: slices.Clone(u.Models)}
}
