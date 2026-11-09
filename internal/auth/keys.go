package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"

	"golang.org/x/crypto/bcrypt"
)

// Above the library default: the users file gets copied around, so an
// offline attack on a leaked copy is the realistic threat. The per-request
// cost is absorbed by the credential cache.
const bcryptCost = 12

// apiKeyPrefix starts every generated API key so secret scanners and humans
// can recognise one.
const apiKeyPrefix = "wh_"

// HashPassword returns a bcrypt hash of password for the users file
// password_hash field.
func HashPassword(password string) (string, error) {
	if password == "" {
		return "", errors.New("empty password")
	}
	h, err := bcrypt.GenerateFromPassword([]byte(password), bcryptCost)
	if err != nil {
		return "", err
	}
	return string(h), nil
}

// NewAPIKey generates a random API key. key is shown to the user once and
// never stored; hash goes into the users file api_keys list.
//
// SHA-256 rather than bcrypt is deliberate: the key carries 256 random bits,
// so a slow hash adds nothing against guessing and would cost every request.
func NewAPIKey() (key, hash string, err error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", "", err
	}
	key = apiKeyPrefix + base64.RawURLEncoding.EncodeToString(b[:])
	return key, hashAPIKey(key), nil
}

// hashAPIKey returns the stored form of key: "sha256:<hex>".
func hashAPIKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return apiKeyHashPrefix + hex.EncodeToString(sum[:])
}
