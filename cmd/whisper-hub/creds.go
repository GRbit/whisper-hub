package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"

	"github.com/GRbit/whisper-hub/internal/auth"
)

// hashCmd reads a password (twice, without echo, on a terminal; one line
// from a pipe otherwise) and prints its bcrypt hash.
func hashCmd(stdin *os.File, stdout, stderr io.Writer) error {
	var pw string
	if term.IsTerminal(int(stdin.Fd())) {
		a, err := prompt(stdin, stderr, "Password: ")
		if err != nil {
			return err
		}
		b, err := prompt(stdin, stderr, "Repeat password: ")
		if err != nil {
			return err
		}
		if a != b {
			return errors.New("passwords do not match")
		}
		pw = a
	} else {
		line, err := bufio.NewReader(stdin).ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		pw = strings.TrimRight(line, "\r\n")
	}
	if pw == "" {
		return errors.New("empty password")
	}
	h, err := auth.HashPassword(pw)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(stdout, "%s\n", h)
	return err
}

func prompt(stdin *os.File, stderr io.Writer, text string) (string, error) {
	fmt.Fprint(stderr, text)
	b, err := term.ReadPassword(int(stdin.Fd()))
	fmt.Fprintln(stderr)
	return string(b), err
}

func apikeyCmd(stdout io.Writer) error {
	key, hash, err := auth.NewAPIKey()
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(stdout, "key:  %s\n      (give it to the user; it is not stored and cannot be shown again)\nhash: %s\n      (add it to the user's api_keys in the users file)\n", key, hash)
	return err
}
