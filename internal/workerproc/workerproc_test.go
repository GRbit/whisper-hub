package workerproc

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

func TestEnvEngineSelection(t *testing.T) {
	c := Config{Engine: "faster_whisper"}
	env, err := c.Env([]string{"PATH=/bin"})
	if err != nil || !slices.Contains(env, "ASR_ENGINE=faster_whisper") {
		t.Fatalf("unset ASR_ENGINE: %v %v", env, err)
	}
	if env, err := c.Env([]string{"ASR_ENGINE=faster_whisper"}); err != nil || len(env) != 1 {
		t.Fatalf("matching ASR_ENGINE: %v %v", env, err)
	}
	_, err = c.Env([]string{"ASR_ENGINE=openai_whisper"})
	if err == nil || !strings.Contains(err.Error(), "use the whisper-hub:openai_whisper or whisper-hub:all image") {
		t.Fatalf("missing engine: %v", err)
	}
	_, err = Config{Engine: "faster_whisper", ImageTag: "whisper-hub:%s-gpu"}.Env([]string{"ASR_ENGINE=whisperx"})
	if err == nil || !strings.Contains(err.Error(), "whisper-hub:whisperx-gpu") {
		t.Fatalf("gpu tag: %v", err)
	}
	if _, err := c.Env([]string{"ASR_ENGINE=vosk"}); err == nil || !strings.Contains(err.Error(), "not an engine") {
		t.Fatalf("unknown engine: %v", err)
	}
	// The all-engines image behaves as upstream: unset means openai_whisper.
	all := Config{Engine: EngineAll}
	if env, err := all.Env(nil); err != nil || !slices.Equal(env, []string{"ASR_ENGINE=openai_whisper"}) {
		t.Fatalf("all, unset: %v %v", env, err)
	}
	for _, e := range []string{"faster_whisper", "whisperx", "openai_whisper"} {
		if _, err := all.Env([]string{"ASR_ENGINE=" + e}); err != nil {
			t.Fatalf("all, %s: %v", e, err)
		}
	}
	if _, err := all.Env([]string{"ASR_ENGINE=fake"}); err == nil {
		t.Fatal("all image accepted the fake engine")
	}
	if env, err := (Config{}).Env([]string{"ASR_ENGINE=anything"}); err != nil || len(env) != 1 {
		t.Fatalf("no engine configured: %v %v", env, err)
	}
}

func TestNoCommandStartsNothing(t *testing.T) {
	p, err := Start(Config{}, nil, quiet)
	if p != nil || err != nil {
		t.Fatalf("%v %v", p, err)
	}
}

func TestExitIsReported(t *testing.T) {
	p, err := Start(Config{Command: []string{"sh", "-c", "exit 3"}}, nil, quiet)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-p.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("exit not reported")
	}
	if p.Err() == nil {
		t.Fatal("non-zero exit reported as success")
	}
	p.Stop() // no-op after exit
}

// The worker must see the environment, with ASR_ENGINE filled in.
func TestEnvReachesWorker(t *testing.T) {
	out := filepath.Join(t.TempDir(), "env")
	p, err := Start(Config{Engine: "fake", Command: []string{"sh", "-c", "echo \"$ASR_ENGINE $ASR_MODEL\" > " + out}},
		[]string{"ASR_MODEL=tiny", "PATH=" + os.Getenv("PATH")}, quiet)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Err(); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(out); strings.TrimSpace(string(b)) != "fake tiny" {
		t.Fatalf("worker saw %q", b)
	}
}

func TestStopTermThenKill(t *testing.T) {
	// Honors SIGTERM: exec makes sleep the process, which dies on TERM.
	p, err := Start(Config{Command: []string{"sh", "-c", "exec sleep 30"}}, nil, quiet)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	p.Stop()
	if time.Since(start) > 3*time.Second {
		t.Fatal("SIGTERM not delivered")
	}

	// Ignores SIGTERM: killed after StopTimeout.
	p, err = Start(Config{Command: []string{"sh", "-c", "trap '' TERM; while :; do sleep 0.05; done"}, StopTimeout: 200 * time.Millisecond}, nil, quiet)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond) // let sh install the trap
	start = time.Now()
	p.Stop()
	if el := time.Since(start); el < 150*time.Millisecond || el > 3*time.Second {
		t.Fatalf("kill after timeout took %v", el)
	}
}

func TestStartFailure(t *testing.T) {
	if _, err := Start(Config{Command: []string{"/nonexistent/worker"}}, nil, quiet); err == nil {
		t.Fatal("missing command started")
	}
}
