package auth

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/crypto/bcrypt"
	"gopkg.in/yaml.v3"
)

// account is one entry of the users file.
//
// Hashes only: the file is meant to be reviewable, and mountable read-only,
// without handing anyone a working login.
type account struct {
	Name string `yaml:"name"`
	// PasswordHash is a bcrypt hash for HTTP Basic. Optional: a user may
	// log in with API keys only, or exist only as the Wyoming identity.
	PasswordHash string `yaml:"password_hash,omitempty"`
	// APIKeys are "sha256:<hex>" hashes of the user's API keys, as printed
	// by NewAPIKey.
	APIKeys []string `yaml:"api_keys,omitempty"`
	// Models are the model id patterns this user may use besides the
	// default model: exact ids or "*" globs.
	Models []string `yaml:"models,omitempty"`
	// Priority, when set, replaces the protocol default of hub.priorities
	// for this user's jobs: higher runs first and may interrupt lower.
	Priority *int `yaml:"priority,omitempty"`
	Admin    bool `yaml:"admin,omitempty"`
	Disabled bool `yaml:"disabled,omitempty"`
}

type usersFile struct {
	Users []account `yaml:"users"`
}

// apiKeyEntry is flattened from all users so a key check can scan every
// key without an early exit.
type apiKeyEntry struct {
	sum  [sha256.Size]byte
	user *account
}

// users is an immutable parsed users file.
type users struct {
	list   []account
	byName map[string]*account
	keys   []apiKeyEntry

	// Hashing the dummy at the cost the file actually uses keeps unknown
	// users as slow as real ones; computed lazily because it costs ~250ms.
	dummyOnce sync.Once
	dummyCost int
	dummy     []byte
}

func (u *users) lookup(name string) (*account, bool) {
	usr, ok := u.byName[name]
	return usr, ok
}

func (u *users) dummyHash() []byte {
	u.dummyOnce.Do(func() {
		h, err := bcrypt.GenerateFromPassword([]byte("whisper-hub dummy password"), u.dummyCost)
		if err != nil {
			// Unreachable for a validated cost; a fixed hash still runs bcrypt.
			h = []byte(fallbackDummyHash)
		}
		u.dummy = h
	})
	return u.dummy
}

// A cost-12 hash of a value nobody knows, used only if generation fails.
const fallbackDummyHash = "$2a$12$C6UzMDM.H6dfI/f/IKcEe.7SLNBn2GBSPMPHnxwFhpO0cOhVe/YRO"

const apiKeyHashPrefix = "sha256:"

func parseUsers(raw []byte) (*users, error) {
	var f usersFile
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	// A misspelled key must fail, not be ignored: "password-hash" would leave
	// the user with an empty hash and no clue why nothing works.
	dec.KnownFields(true)
	// An empty file decodes to io.EOF; treat it as zero users.
	if err := dec.Decode(&f); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse users file: %w", err)
	}

	u := &users{list: f.Users, byName: make(map[string]*account, len(f.Users)), dummyCost: bcryptCost}
	seenKeys := make(map[[sha256.Size]byte]string)
	maxCost := 0
	for i := range u.list {
		usr := &u.list[i]
		if usr.Name == "" {
			return nil, fmt.Errorf("user %d has an empty name", i)
		}
		if _, dup := u.byName[usr.Name]; dup {
			return nil, fmt.Errorf("user %q is defined more than once", usr.Name)
		}
		if usr.PasswordHash != "" {
			// Catches a plaintext password pasted here, which would otherwise
			// fail every login silently.
			cost, err := bcrypt.Cost([]byte(usr.PasswordHash))
			if err != nil {
				return nil, fmt.Errorf("user %q has an invalid bcrypt password_hash: %w", usr.Name, err)
			}
			maxCost = max(maxCost, cost)
		}
		for _, k := range usr.APIKeys {
			sum, err := parseKeyHash(k)
			if err != nil {
				return nil, fmt.Errorf("user %q: %w", usr.Name, err)
			}
			if other, dup := seenKeys[sum]; dup {
				return nil, fmt.Errorf("user %q has an api key already used by %q", usr.Name, other)
			}
			seenKeys[sum] = usr.Name
			u.keys = append(u.keys, apiKeyEntry{sum: sum, user: usr})
		}
		for _, p := range usr.Models {
			if strings.TrimSpace(p) == "" {
				return nil, fmt.Errorf("user %q has an empty models pattern", usr.Name)
			}
		}
		u.byName[usr.Name] = usr
	}
	if maxCost > 0 {
		u.dummyCost = maxCost
	}
	return u, nil
}

func parseKeyHash(s string) ([sha256.Size]byte, error) {
	var sum [sha256.Size]byte
	hexPart, ok := strings.CutPrefix(s, apiKeyHashPrefix)
	if !ok {
		return sum, errors.New(`api key hash must look like "sha256:<64 hex digits>" (store the hash, not the key)`)
	}
	b, err := hex.DecodeString(hexPart)
	if err != nil || len(b) != sha256.Size {
		return sum, errors.New(`api key hash must look like "sha256:<64 hex digits>"`)
	}
	copy(sum[:], b)
	return sum, nil
}

const usersTemplate = `# whisper-hub users.
#
# This file was created automatically because it did not exist. Until it names
# at least one user, every request to a protected path is rejected with 401.
#
# Passwords and API keys are never stored, only their hashes:
#   whisper-hub hash      prints a bcrypt hash for password_hash
#   whisper-hub apikey    prints a new API key once, and its hash for api_keys
#
# Changes are picked up automatically within a few seconds; no restart needed.
#
# users:
#   - name: alice
#     password_hash: "$2a$12$..."
#     api_keys: ["sha256:..."]
#     models: ["small", "large-*"]   # besides the default model; "*" = all
#     priority: 80                   # instead of the protocol default (hub.priorities)
#     admin: true                    # may read /_hub/usage
#     disabled: false

users: []
`

// Comparable, so one == covers both contents and error.
type fileState struct {
	sum     [32]byte
	errText string
}

// store holds the active users and swaps them atomically.
//
// No users file problem is fatal: a gateway that will not start over a typo
// cannot be exec'd into and fixed. It comes up rejecting everyone instead.
type store struct {
	path string

	mu   sync.RWMutex
	cur  *users
	last fileState

	// Drops the credential cache, which would otherwise keep honouring a
	// password just removed from the file.
	onReload func(*users)
}

func newStore(path string, onReload func(*users)) *store {
	s := &store{
		path:     path,
		cur:      &users{byName: map[string]*account{}, dummyCost: bcryptCost},
		onReload: onReload,
	}
	_ = s.refresh(true)
	return s
}

func (s *store) get() *users {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cur
}

// refresh re-reads the file. Unchanged content is skipped unless forced, so
// a persistently broken file logs once rather than every poll.
func (s *store) refresh(force bool) error {
	raw, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		if s.writeTemplate() {
			raw, err = os.ReadFile(s.path)
		}
	}

	state := fileState{}
	if err != nil {
		state.errText = err.Error()
	} else {
		state.sum = sha256.Sum256(raw)
	}

	s.mu.Lock()
	unchanged := state == s.last
	s.last = state
	s.mu.Unlock()
	if unchanged && !force {
		return nil
	}

	if err != nil {
		slog.Error("auth: cannot read users file, keeping the users already loaded",
			"path", s.path, "error", err)
		return fmt.Errorf("read users file: %w", err)
	}

	u, err := parseUsers(raw)
	if err != nil {
		slog.Error("auth: users file is invalid, keeping the users already loaded",
			"path", s.path, "error", err)
		return err
	}

	s.mu.Lock()
	s.cur = u
	s.mu.Unlock()

	if s.onReload != nil {
		s.onReload(u)
	}

	if len(u.list) == 0 {
		slog.Warn("auth: no users are configured, every request to a protected path will be rejected; "+
			"add a user to the users file to allow access", "path", s.path)
		return nil
	}
	slog.Info("auth: users loaded", "path", s.path, "users", len(u.list), "api_keys", len(u.keys))
	return nil
}

// Reports whether the file now exists. Failure is fine: a read-only mount is
// a reasonable way to run this, it just means the operator supplies the file.
func (s *store) writeTemplate() bool {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		slog.Warn("auth: users file is missing and its directory could not be created",
			"path", s.path, "error", err)
		return false
	}
	// O_EXCL so a file that appeared since the read is never clobbered.
	f, err := os.OpenFile(s.path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		if errors.Is(err, fs.ErrExist) {
			return true
		}
		slog.Warn("auth: users file is missing and could not be created; supply it yourself, or mount it read-write",
			"path", s.path, "error", err)
		return false
	}
	defer f.Close()

	if _, err := f.WriteString(usersTemplate); err != nil {
		slog.Warn("auth: could not write the users file template", "path", s.path, "error", err)
		return false
	}
	slog.Warn("auth: users file did not exist, created an empty one; edit it to add users, changes are picked up automatically",
		"path", s.path)
	return true
}
