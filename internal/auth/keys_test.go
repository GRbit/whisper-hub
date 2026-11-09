package auth

import (
	"strings"
	"testing"

	"golang.org/x/crypto/bcrypt"
)

func TestHashPasswordLogsIn(t *testing.T) {
	h, err := HashPassword("pw")
	if err != nil {
		t.Fatal(err)
	}
	if cost, _ := bcrypt.Cost([]byte(h)); cost != 12 {
		t.Errorf("cost = %d, want 12", cost)
	}
	a := newAuth(t, Config{UsersFile: writeFile(t, "users:\n  - name: a\n    password_hash: \""+h+"\"\n"), PollInterval: -1})
	if userOf(a, basicReq("a", "pw")) != "a" {
		t.Error("HashPassword output does not log in")
	}
	if _, err := HashPassword(""); err == nil {
		t.Error("empty password accepted")
	}
}

func TestNewAPIKeyLogsIn(t *testing.T) {
	k1, h1, err := NewAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	k2, h2, _ := NewAPIKey()
	if k1 == k2 || h1 == h2 {
		t.Error("two generated keys are equal")
	}
	if !strings.HasPrefix(k1, "wh_") || len(k1) != len("wh_")+43 {
		t.Errorf("key %q has an unexpected shape", k1)
	}
	if h1 != keyHash(k1) {
		t.Errorf("hash %q is not the documented sha256:<hex> of the key", h1)
	}
	a := newAuth(t, Config{UsersFile: writeFile(t, "users:\n  - name: a\n    api_keys: [\""+h1+"\"]\n"), PollInterval: -1})
	if userOf(a, keyReq(k1)) != "a" || userOf(a, keyReq(k2)) != "" {
		t.Error("NewAPIKey key does not authenticate exactly its own user")
	}
}
