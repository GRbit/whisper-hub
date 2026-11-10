package auth

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestMinimalUsersFileWorks(t *testing.T) {
	// Names and password hashes only: every other field is optional.
	body := "# Credentials.\n" +
		"users:\n" +
		"  - name: alice\n" +
		"    password_hash: \"" + hashOf(t, "secret") + "\"\n" +
		"  - name: bob\n" +
		"    password_hash: \"" + hashOf(t, "hunter2") + "\"\n"
	a := newAuth(t, Config{UsersFile: writeFile(t, body), PollInterval: -1})
	if userOf(a, basicReq("alice", "secret")) != "alice" || userOf(a, basicReq("bob", "hunter2")) != "bob" {
		t.Error("a users file with names and password hashes only does not authenticate")
	}
}

func TestBadUsersFileFailsClosedAndSaysWhy(t *testing.T) {
	valid := hashOf(t, "pw")
	cases := []struct {
		name string
		body string
		want string
	}{
		{"empty name", "users:\n  - name: \"\"\n    password_hash: \"" + valid + "\"\n", "empty name"},
		{"duplicate names", "users:\n  - name: alice\n  - name: alice\n", "more than once"},
		// The most likely operator mistake: pasting the password itself.
		{"plaintext password", "users:\n  - name: alice\n    password_hash: \"secret\"\n", "invalid bcrypt"},
		{"misspelled key", "users:\n  - name: alice\n    password-hash: \"x\"\n", "field password-hash not found"},
		{"not yaml at all", "\tthis is not yaml", "parse users file"},
		{"plaintext api key", "users:\n  - name: alice\n    api_keys: [\"" + aliceKey + "\"]\n", "store the hash"},
		{"short api key hash", "users:\n  - name: alice\n    api_keys: [\"sha256:abcd\"]\n", "64 hex"},
		{"same key twice", "users:\n  - name: a\n    api_keys: [\"" + keyHash(aliceKey) + "\"]\n" +
			"  - name: b\n    api_keys: [\"" + keyHash(aliceKey) + "\"]\n", "already used"},
		{"empty model pattern", "users:\n  - name: a\n    models: [\"\"]\n", "empty models pattern"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logs := captureLogs(t)
			// A broken file never stops startup: a gateway that will not start
			// cannot be exec'd into and repaired.
			a := newAuth(t, Config{UsersFile: writeFile(t, tc.body), PollInterval: -1})
			if u := userOf(a, keyReq(aliceKey)); u != "" {
				t.Errorf("request accepted as %q from a broken file", u)
			}
			if !strings.Contains(logs.String(), tc.want) {
				t.Errorf("log does not mention %q:\n%s", tc.want, logs.String())
			}
			if err := a.Reload(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Reload error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestEmptyUsersFileRejectsEveryone(t *testing.T) {
	// Allowed so the service stays up, but it must fail closed, not open.
	for _, body := range []string{"users: []\n", ""} {
		a := newAuth(t, Config{UsersFile: writeFile(t, body), PollInterval: -1})
		if err := a.Reload(); err != nil {
			t.Errorf("Reload(%q) = %v, want no error", body, err)
		}
		for _, r := range []*http.Request{basicReq("alice", "secret"), basicReq("", ""), keyReq("anything")} {
			if u := userOf(a, r); u != "" {
				t.Errorf("accepted %q with no users configured", u)
			}
		}
	}
}

func TestMissingFileCreatesTemplate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "users.yaml")
	a := newAuth(t, Config{UsersFile: path, PollInterval: -1})
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("template was not created: %v", err)
	}
	// The template must load as is, or the next poll would report it.
	if err := a.Reload(); err != nil {
		t.Errorf("generated template does not load: %v", err)
	}
	for _, want := range []string{"whisper-hub hash", "whisper-hub apikey"} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("template does not mention %q", want)
		}
	}
	if u := userOf(a, basicReq("alice", "secret")); u != "" {
		t.Error("template must reject everyone")
	}
}

func TestExistingFileIsNeverOverwritten(t *testing.T) {
	body := validUsers(t)
	path := writeFile(t, body)
	newAuth(t, Config{UsersFile: path, PollInterval: -1})
	if raw, _ := os.ReadFile(path); string(raw) != body {
		t.Error("an existing users file was modified")
	}
}

func TestUnreadablePathDoesNotStopStartup(t *testing.T) {
	// A directory where the file should be is what Docker leaves behind when
	// a bind-mounted path does not exist on the host.
	dir := filepath.Join(t.TempDir(), "users.yaml")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	a := newAuth(t, Config{UsersFile: dir, PollInterval: -1})
	if w := serve(a.Middleware(echoIdentity), basicReq("alice", "secret")); w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", w.Code)
	}
}

func TestReloadKeepsPreviousUsersOnBadEdit(t *testing.T) {
	// A typo saved into the users file of a running service must not lock
	// every user out.
	path := writeFile(t, validUsers(t))
	a := newAuth(t, Config{UsersFile: path, PollInterval: -1})
	rewrite(t, path, "users:\n  - name: a\n    password_hash: \"not-a-hash\"\n")
	if err := a.Reload(); err == nil {
		t.Error("Reload of a broken file returned no error")
	}
	if userOf(a, basicReq("alice", "secret")) != "alice" {
		t.Error("alice was locked out by a failed reload")
	}
}

func TestReloadRevokesAtOnce(t *testing.T) {
	// The verified-credentials cache must not keep a removed user alive.
	path := writeFile(t, validUsers(t))
	a := newAuth(t, Config{UsersFile: path, PollInterval: -1})
	if userOf(a, basicReq("alice", "secret")) != "alice" {
		t.Fatal("alice should authenticate initially")
	}
	rewrite(t, path, "users:\n  - name: bob\n    password_hash: \""+hashOf(t, "hunter2")+"\"\n")
	if err := a.Reload(); err != nil {
		t.Fatal(err)
	}
	if u := userOf(a, basicReq("alice", "secret")); u != "" {
		t.Error("a removed user is still authenticated by password")
	}
	if u := userOf(a, keyReq(aliceKey)); u != "" {
		t.Error("a removed user's api key still works")
	}
	if userOf(a, basicReq("bob", "hunter2")) != "bob" {
		t.Error("remaining user stopped working after reload")
	}
}

func TestPolledPasswordChangeStopsCachedLogin(t *testing.T) {
	// alice's old password is in the credential cache; once the poller reads
	// the edited file it must stop working with no explicit Reload.
	path := writeFile(t, validUsers(t))
	a := newAuth(t, Config{UsersFile: path, PollInterval: 5 * time.Millisecond})
	go a.Run(t.Context())
	if userOf(a, basicReq("alice", "secret")) != "alice" {
		t.Fatal("alice should authenticate initially")
	}
	rewrite(t, path, strings.Replace(validUsers(t), "alice\n    password_hash: \"", "alice\n    password_hash: \""+hashOf(t, "new")+"\"\n    #", 1))
	waitFor(t, "the new password to apply", func() bool {
		return userOf(a, basicReq("alice", "new")) == "alice"
	})
	if u := userOf(a, basicReq("alice", "secret")); u != "" {
		t.Error("the old password still works after a change")
	}
}

func TestRunPicksUpEditsWithoutASignal(t *testing.T) {
	// Editing the file inside a running container has to be enough.
	path := writeFile(t, "this is not valid yaml: [")
	a := newAuth(t, Config{UsersFile: path, PollInterval: 5 * time.Millisecond})
	go a.Run(t.Context())
	rewrite(t, path, validUsers(t))
	waitFor(t, "alice to appear", func() bool { return userOf(a, keyReq(aliceKey)) == "alice" })
}

func TestReloadWhileAuthenticatingIsRaceFree(t *testing.T) {
	// Run with -race: requests read the users snapshot while Reload swaps it
	// and clears the credential cache. Broken edits keep the old users, so
	// alice must never be rejected.
	path := writeFile(t, validUsers(t))
	a := newAuth(t, Config{UsersFile: path, PollInterval: -1, RateLimit: RateLimitConfig{Failures: -1}})
	good := validUsers(t)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for range 4 {
		wg.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}
				if userOf(a, basicReq("alice", "secret")) != "alice" || userOf(a, keyReq(aliceKey)) != "alice" {
					t.Error("alice rejected during reload")
					return
				}
			}
		})
	}
	for i := range 50 {
		if i%2 == 0 {
			rewrite(t, path, good)
		} else {
			rewrite(t, path, good+"  - name: \"\"\n")
		}
		_ = a.Reload()
	}
	close(stop)
	wg.Wait()
}
