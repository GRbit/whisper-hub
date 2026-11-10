package auth

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GRbit/whisper-hub/internal/core"
)

// seedTotals leaves a totals file at path holding one job of alice with
// the given audio seconds, written by an Auth that has since shut down.
func seedTotals(t *testing.T, path string, seconds float64) {
	t.Helper()
	a, err := New(Config{Usage: UsageConfig{TotalsFile: path}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", seconds))
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func fileSize(path string) int64 {
	st, err := os.Stat(path)
	if err != nil {
		return -1
	}
	return st.Size()
}

func TestDefaultTotalsFilePersists(t *testing.T) {
	def := filepath.Join(t.TempDir(), "missing", "dir", "usage.json")
	a, err := New(Config{Usage: UsageConfig{DefaultTotalsFile: def}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", 3))
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if fileSize(def) <= 0 {
		t.Fatalf("default totals file not written")
	}
	b := usageAuth(t, UsageConfig{DefaultTotalsFile: def})
	if got := b.Totals()["alice"]; got.Requests != 1 || got.AudioSeconds != 3 {
		t.Errorf("after restart: %+v", got)
	}
}

// The image default must never stop the service: it degrades to memory.
func TestUnwritableDefaultTotalsFileStaysInMemory(t *testing.T) {
	logs := captureLogs(t)
	// A regular file as the parent directory fails even for root.
	blocker := writeFile(t, "")
	def := filepath.Join(blocker, "usage.json")
	a, err := New(Config{Usage: UsageConfig{DefaultTotalsFile: def}}, nil)
	if err != nil {
		t.Fatalf("New failed on an unwritable default: %v", err)
	}
	a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", 3))
	if err := a.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if a.Totals()["alice"].Requests != 1 {
		t.Errorf("totals not kept in memory")
	}
	if n := strings.Count(logs.String(), "level=WARN"); n != 1 {
		t.Errorf("want exactly one warning, got %d:\n%s", n, logs)
	}
	if !strings.Contains(logs.String(), "memory") {
		t.Errorf("warning does not say totals stay in memory:\n%s", logs)
	}
}

// An explicitly configured path keeps failing startup, as before.
func TestUnwritableTotalsFileIsFatal(t *testing.T) {
	blocker := writeFile(t, "")
	if _, err := New(Config{Usage: UsageConfig{TotalsFile: filepath.Join(blocker, "usage.json")}}, nil); err == nil {
		t.Fatal("New accepted an unwritable totals_file")
	}
}

func TestTotalsMoveFromDefaultFile(t *testing.T) {
	for name, prepare := range map[string]func(t *testing.T, path string){
		"configured missing": func(*testing.T, string) {},
		"configured empty": func(t *testing.T, path string) {
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, nil, 0o600); err != nil {
				t.Fatal(err)
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			def := filepath.Join(t.TempDir(), "usage.json")
			seedTotals(t, def, 3)
			conf := filepath.Join(t.TempDir(), "new", "usage.json")
			prepare(t, conf)

			a, err := New(Config{Usage: UsageConfig{TotalsFile: conf, DefaultTotalsFile: def}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := a.Totals()["alice"]; got.Requests != 1 || got.AudioSeconds != 3 {
				t.Errorf("moved totals = %+v", got)
			}
			if fileSize(conf) <= 0 {
				t.Errorf("configured file not written before the old one was removed")
			}
			if _, err := os.Stat(def); !os.IsNotExist(err) {
				t.Errorf("old file still there: %v", err)
			}
			var moved []string
			for _, l := range strings.Split(logs.String(), "\n") {
				if strings.Contains(l, "level=INFO") && strings.Contains(l, "moved") {
					moved = append(moved, l)
				}
			}
			if len(moved) != 1 || !strings.Contains(moved[0], def) || !strings.Contains(moved[0], conf) {
				t.Errorf("want one Info line naming both paths, got %q", moved)
			}
			a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", 4))
			if err := a.Close(); err != nil {
				t.Fatal(err)
			}
			b := usageAuth(t, UsageConfig{TotalsFile: conf, DefaultTotalsFile: def})
			if got := b.Totals()["alice"]; got.Requests != 2 || got.AudioSeconds != 7 {
				t.Errorf("counters did not continue: %+v", got)
			}
		})
	}
}

func TestTotalsNotMovedOntoExistingFile(t *testing.T) {
	def := filepath.Join(t.TempDir(), "usage.json")
	seedTotals(t, def, 3)
	conf := filepath.Join(t.TempDir(), "usage.json")
	seedTotals(t, conf, 10)
	before, _ := os.ReadFile(def)

	a := usageAuth(t, UsageConfig{TotalsFile: conf, DefaultTotalsFile: def})
	if got := a.Totals()["alice"]; got.Requests != 1 || got.AudioSeconds != 10 {
		t.Errorf("totals = %+v, want the configured file's", got)
	}
	if after, _ := os.ReadFile(def); !bytes.Equal(before, after) {
		t.Errorf("old file changed")
	}
}

func TestTotalsNotMovedOntoItself(t *testing.T) {
	dir := t.TempDir()
	def := filepath.Join(dir, "usage.json")
	seedTotals(t, def, 3)
	a := usageAuth(t, UsageConfig{TotalsFile: dir + "/./usage.json", DefaultTotalsFile: def})
	if got := a.Totals()["alice"]; got.Requests != 1 {
		t.Errorf("totals = %+v", got)
	}
	if fileSize(def) <= 0 {
		t.Errorf("the file was removed")
	}
}

// A failed migration loses nothing and does not stop startup.
func TestFailedMoveKeepsOldFile(t *testing.T) {
	t.Run("unreadable totals", func(t *testing.T) {
		logs := captureLogs(t)
		def := filepath.Join(t.TempDir(), "usage.json")
		if err := os.WriteFile(def, []byte("{not json"), 0o600); err != nil {
			t.Fatal(err)
		}
		conf := filepath.Join(t.TempDir(), "usage.json")
		usageAuth(t, UsageConfig{TotalsFile: conf, DefaultTotalsFile: def})
		if b, _ := os.ReadFile(def); string(b) != "{not json" {
			t.Errorf("old file changed: %q", b)
		}
		if !strings.Contains(logs.String(), "level=WARN") {
			t.Errorf("no warning:\n%s", logs)
		}
	})
	t.Run("unwritable target", func(t *testing.T) {
		if os.Geteuid() == 0 {
			t.Skip("root ignores directory permissions")
		}
		logs := captureLogs(t)
		def := filepath.Join(t.TempDir(), "usage.json")
		seedTotals(t, def, 3)
		confDir := t.TempDir()
		if err := os.Chmod(confDir, 0o555); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.Chmod(confDir, 0o755) })
		a, err := New(Config{Usage: UsageConfig{TotalsFile: filepath.Join(confDir, "usage.json"), DefaultTotalsFile: def}}, nil)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer a.Close()
		if got := a.Totals()["alice"]; got.Requests != 1 || got.AudioSeconds != 3 {
			t.Errorf("loaded totals = %+v, want the old file's", got)
		}
		if fileSize(def) <= 0 {
			t.Errorf("old file removed although the new one was not written")
		}
		if !strings.Contains(logs.String(), "level=WARN") {
			t.Errorf("no warning:\n%s", logs)
		}
	})
}
