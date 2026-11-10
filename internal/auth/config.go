// Package auth implements whisper-hub authentication, per-user model access
// and usage accounting.
//
// It is an in-process library: Auth.Middleware guards the HTTP mux, Auth is
// the core.Authorizer and core.UsageRecorder handed to the hub, and
// Auth.WyomingIdentity maps Wyoming connections to a configured user.
package auth

import (
	"fmt"
	"net/netip"
	"slices"
	"time"
)

// Config configures authentication, model access and usage accounting.
//
// Zero values mean the documented default; see Resolved.
type Config struct {
	// UsersFile is the users file. Empty (the default) turns authentication
	// off: every request has admin access, trusted because the network
	// restricts who reaches the service, as with upstream
	// whisper-asr-webservice.
	// A configured but missing file is created as an empty template and
	// rejects everyone until it names a user.
	UsersFile string `yaml:"users_file"`
	// PollInterval is how often UsersFile is checked for edits. Default 5s;
	// negative disables polling (Reload still works).
	PollInterval time.Duration `yaml:"poll_interval"`
	// Realm is shown in the browser login prompt on legacy paths.
	// Default "whisper".
	Realm string `yaml:"realm"`
	// ProtectedPaths need credentials; everything else is public. An entry
	// matches itself and everything below it ("/v1/" also matches "/v1").
	// Default ["/asr", "/detect-language", "/v1/"].
	ProtectedPaths []string `yaml:"protected_paths"`
	// TrustForwarded makes the client address come from X-Forwarded-For or
	// X-Real-IP. Only enable behind a trusted reverse proxy, otherwise
	// clients choose their own address for rate limiting and logs.
	TrustForwarded bool `yaml:"trust_forwarded"`

	RateLimit RateLimitConfig `yaml:"rate_limit"`
	Wyoming   WyomingConfig   `yaml:"wyoming"`
	Usage     UsageConfig     `yaml:"usage"`
}

// RateLimitConfig throttles failed authentication attempts per client
// address. A blocked address gets 401 with a Retry-After header (not 429)
// without a bcrypt evaluation; cached credentials and valid API keys still
// pass, since neither costs bcrypt time.
type RateLimitConfig struct {
	// Failures allowed per Window before the address is blocked for the
	// rest of the window. Default 10; negative disables rate limiting.
	Failures int `yaml:"failures"`
	// Window is the counting period. Default 1m.
	Window time.Duration `yaml:"window"`
}

// WyomingConfig maps Wyoming connections, which carry no credentials, to a
// user.
type WyomingConfig struct {
	// User is the users file entry Wyoming connections act as; its models
	// patterns and disabled flag apply. With authentication on and User
	// empty, Wyoming connections are refused. With authentication off it
	// only names the caller in usage records (default "admin").
	User string `yaml:"user"`
	// AllowCIDRs restricts source addresses ("192.168.1.0/24", "::1/128").
	// Empty (the default) allows every address.
	AllowCIDRs []string `yaml:"allow_cidrs"`
}

// UsageConfig configures usage accounting.
type UsageConfig struct {
	// TotalsFile persists per-user totals across restarts. It is rewritten
	// atomically every SnapshotInterval when something changed, and on
	// Close. Empty (the default) falls back to DefaultTotalsFile.
	TotalsFile string `yaml:"totals_file"`
	// DefaultTotalsFile is where totals persist while TotalsFile is empty;
	// the images set it to /data/usage.json. Unlike TotalsFile it is best
	// effort: an unwritable location keeps totals in memory with a warning.
	// While TotalsFile does not exist or is empty, a non-empty
	// DefaultTotalsFile is moved there. Default "": none.
	DefaultTotalsFile string `yaml:"default_totals_file"`
	// SnapshotInterval is how often TotalsFile is written. Default 30s.
	SnapshotInterval time.Duration `yaml:"snapshot_interval"`
	// LogFile receives one JSON line per job and per rejected login.
	// Empty (the default) disables the request log.
	LogFile string `yaml:"log_file"`
	// LogMaxBytes rotates LogFile to LogFile.1 (shifting older ones up)
	// once it would exceed this size. Default 0: never rotate.
	LogMaxBytes int64 `yaml:"log_max_bytes"`
	// LogMaxBackups is how many rotated files are kept. Default 5.
	LogMaxBackups int `yaml:"log_max_backups"`
}

const (
	defaultPollInterval     = 5 * time.Second
	defaultRealm            = "whisper"
	defaultRateFailures     = 10
	defaultRateWindow       = time.Minute
	defaultSnapshotInterval = 30 * time.Second
	defaultLogMaxBackups    = 5
)

// defaultProtectedPaths is the path policy used when Config.ProtectedPaths
// is empty.
var defaultProtectedPaths = []string{"/asr", "/detect-language", "/v1/"}

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it.
func (c Config) Resolved() Config {
	if c.PollInterval == 0 {
		c.PollInterval = defaultPollInterval
	}
	if c.Realm == "" {
		c.Realm = defaultRealm
	}
	if len(c.ProtectedPaths) == 0 {
		c.ProtectedPaths = slices.Clone(defaultProtectedPaths)
	}
	if c.RateLimit.Failures == 0 {
		c.RateLimit.Failures = defaultRateFailures
	}
	if c.RateLimit.Window <= 0 {
		c.RateLimit.Window = defaultRateWindow
	}
	c.Usage = c.Usage.resolved()
	return c
}

func (c UsageConfig) resolved() UsageConfig {
	if c.SnapshotInterval <= 0 {
		c.SnapshotInterval = defaultSnapshotInterval
	}
	if c.LogMaxBackups <= 0 {
		c.LogMaxBackups = defaultLogMaxBackups
	}
	return c
}

func parseCIDRs(list []string) ([]netip.Prefix, error) {
	out := make([]netip.Prefix, 0, len(list))
	for _, s := range list {
		p, err := netip.ParsePrefix(s)
		if err != nil {
			// A bare address is a common way to write "only this host".
			a, aerr := netip.ParseAddr(s)
			if aerr != nil {
				return nil, fmt.Errorf("wyoming.allow_cidrs: %q is neither a CIDR nor an address", s)
			}
			p = netip.PrefixFrom(a, a.BitLen())
		}
		out = append(out, p.Masked())
	}
	return out, nil
}
