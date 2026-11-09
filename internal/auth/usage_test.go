package auth

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

func okEvent(user, protocol, model string, seconds float64) core.UsageEvent {
	return core.UsageEvent{
		Time: time.Unix(1700000000, 0), User: user, Protocol: protocol, Endpoint: "/asr",
		Model: model, Engine: "faster_whisper", Task: core.TaskTranscribe, AudioSeconds: seconds,
		Queued: 2 * time.Second, Elapsed: 5 * time.Second, Status: "ok",
	}
}

func usageAuth(t *testing.T, u UsageConfig) *Auth {
	t.Helper()
	return newAuth(t, Config{Usage: u})
}

func TestTotalsAccounting(t *testing.T) {
	a := usageAuth(t, UsageConfig{})
	ctx := context.Background()
	a.Record(ctx, okEvent("alice", core.ProtocolLegacy, "tiny", 10))
	a.Record(ctx, okEvent("alice", core.ProtocolOpenAI, "small", 5))
	failed := okEvent("alice", core.ProtocolOpenAI, "small", 7)
	failed.Status, failed.Error = "internal", "worker died"
	a.Record(ctx, failed)
	undecoded := okEvent("alice", core.ProtocolLegacy, "", 0)
	undecoded.Status = "bad_audio"
	a.Record(ctx, undecoded)

	got := a.Totals()["alice"]
	if got.Requests != 4 || got.OKRequests != 2 || got.FailedRequests != 2 {
		t.Errorf("counts = %+v", got)
	}
	// Failures are never billed: 7s of failed audio stays out of audio_seconds.
	if got.AudioSeconds != 15 || got.FailedSeconds != 7 || got.UnknownRequests != 1 {
		t.Errorf("seconds: billable %v failed %v unknown %d", got.AudioSeconds, got.FailedSeconds, got.UnknownRequests)
	}
	if got.ProcessingSeconds != 12 {
		t.Errorf("processing seconds = %v, want 12 (4 jobs x 3s)", got.ProcessingSeconds)
	}
	if c := got.ByProtocol[core.ProtocolOpenAI]; c.Requests != 2 || c.OKRequests != 1 || c.AudioSeconds != 5 {
		t.Errorf("by_protocol openai = %+v", c)
	}
	if c := got.ByModel["small"]; c.Requests != 2 || c.FailedRequests != 1 {
		t.Errorf("by_model small = %+v", c)
	}
	if c := got.ByModel["unknown"]; c.Requests != 1 {
		t.Errorf("unresolved model bucket = %+v", c)
	}
	if !got.FirstSeen.Equal(time.Unix(1700000000, 0)) {
		t.Errorf("first_seen = %v", got.FirstSeen)
	}

	// Totals returns a copy: mutating it must not reach the recorder.
	snap := a.Totals()
	snap["alice"].ByModel["tiny"] = Counter{Requests: 999}
	if a.Totals()["alice"].ByModel["tiny"].Requests != 1 {
		t.Error("Totals leaked internal state")
	}
}

func TestEmptyUserIsBilledAsAnonymous(t *testing.T) {
	a := usageAuth(t, UsageConfig{})
	a.Record(context.Background(), okEvent("", core.ProtocolWyoming, "tiny", 1))
	if a.Totals()["anonymous"].Requests != 1 {
		t.Errorf("totals = %+v", a.Totals())
	}
}

func TestTotalsSurviveRestart(t *testing.T) {
	file := filepath.Join(t.TempDir(), "usage", "totals.json")
	a, err := New(Config{Usage: UsageConfig{TotalsFile: file}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	a.Record(context.Background(), okEvent("alice", core.ProtocolWyoming, "tiny", 3))
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	b := usageAuth(t, UsageConfig{TotalsFile: file})
	b.Record(context.Background(), okEvent("alice", core.ProtocolWyoming, "tiny", 4))
	got := b.Totals()["alice"]
	if got.Requests != 2 || got.AudioSeconds != 7 || got.ByProtocol[core.ProtocolWyoming].Requests != 2 {
		t.Errorf("after restart: %+v", got)
	}
}

func TestOldSidecarSnapshotLoads(t *testing.T) {
	// Verbatim from docs/auth-proxy.md: counters must carry over.
	old := `{
  "alice": {
    "requests": 42,
    "ok_requests": 40,
    "failed_requests": 2,
    "audio_seconds": 8130.5,
    "failed_seconds": 300.0,
    "unknown_requests": 1,
    "bytes_in": 918273645,
    "first_seen": "2026-08-01T09:12:03Z",
    "last_seen": "2026-08-04T17:45:11Z"
  }
}`
	file := filepath.Join(t.TempDir(), "usage.json")
	if err := os.WriteFile(file, []byte(old), 0o600); err != nil {
		t.Fatal(err)
	}
	a := usageAuth(t, UsageConfig{TotalsFile: file})
	a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", 1.5))
	got := a.Totals()["alice"]
	if got.Requests != 43 || got.AudioSeconds != 8132 || got.BytesIn != 918273645 || got.UnknownRequests != 1 {
		t.Errorf("merged totals = %+v", got)
	}
	if !got.FirstSeen.Equal(time.Date(2026, 8, 1, 9, 12, 3, 0, time.UTC)) {
		t.Errorf("first_seen = %v", got.FirstSeen)
	}
}

func TestCorruptSnapshotIsNotFatal(t *testing.T) {
	file := filepath.Join(t.TempDir(), "usage.json")
	if err := os.WriteFile(file, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if n := len(usageAuth(t, UsageConfig{TotalsFile: file}).Totals()); n != 0 {
		t.Errorf("corrupt snapshot gave %d users, want a fresh start", n)
	}
}

func TestUnchangedTotalsAreNotRewritten(t *testing.T) {
	// A periodic snapshot of nothing new would only wear the disk.
	file := filepath.Join(t.TempDir(), "usage.json")
	a, err := New(Config{Usage: UsageConfig{TotalsFile: file}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Error("an unchanged recorder wrote a snapshot")
	}
}

func TestPeriodicSnapshot(t *testing.T) {
	file := filepath.Join(t.TempDir(), "usage.json")
	a := usageAuth(t, UsageConfig{TotalsFile: file, SnapshotInterval: 5 * time.Millisecond})
	go a.Run(t.Context())
	a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", 1))
	waitFor(t, "a snapshot on disk", func() bool {
		raw, err := os.ReadFile(file)
		return err == nil && strings.Contains(string(raw), `"alice"`)
	})
}

func TestRequestLogHasAllEventFields(t *testing.T) {
	logFile := filepath.Join(t.TempDir(), "requests.jsonl")
	a := usageAuth(t, UsageConfig{LogFile: logFile})
	ev := okEvent("alice", core.ProtocolOpenAI, "small", 12.5)
	ev.Endpoint, ev.Task, ev.Status, ev.Error = "/v1/audio/translations", core.TaskTranslate, "canceled", "client went away"
	a.Record(context.Background(), ev)

	lines := readLines(t, logFile)
	if len(lines) != 1 {
		t.Fatalf("got %d lines", len(lines))
	}
	want := logLine{
		"ts": "2023-11-14T22:13:20Z", "user": "alice", "protocol": "openai", "endpoint": "/v1/audio/translations",
		"model": "small", "engine": "faster_whisper", "task": "translate", "audio_seconds": 12.5,
		"queued_ms": 2000.0, "elapsed_ms": 5000.0, "status": "canceled", "error": "client went away",
	}
	for k, v := range want {
		if lines[0][k] != v {
			t.Errorf("%s = %v, want %v", k, lines[0][k], v)
		}
	}
}

func TestJobLogLine(t *testing.T) {
	logs := captureLogs(t)
	a := usageAuth(t, UsageConfig{})
	a.Record(context.Background(), okEvent("alice", core.ProtocolLegacy, "tiny", 2))
	out := logs.String()
	for _, want := range []string{"level=INFO", `msg="job finished"`, "user=alice", "model=tiny", "status=ok", "audio_seconds=2"} {
		if !strings.Contains(out, want) {
			t.Errorf("job log line lacks %q: %s", want, out)
		}
	}
}

func TestRequestLogRotation(t *testing.T) {
	logFile := filepath.Join(t.TempDir(), "requests.jsonl")
	a := usageAuth(t, UsageConfig{LogFile: logFile, LogMaxBytes: 600, LogMaxBackups: 2})
	for i := range 20 {
		a.Record(context.Background(), okEvent("u"+strconv.Itoa(i), "legacy", "tiny", 1))
	}
	a.Close()
	total := 0
	for _, p := range []string{logFile, logFile + ".1", logFile + ".2"} {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatalf("%s missing: %v", p, err)
		}
		if st.Size() > 600 {
			t.Errorf("%s is %d bytes, over the 600 limit", p, st.Size())
		}
		total += len(readLines(t, p))
	}
	if _, err := os.Stat(logFile + ".3"); !os.IsNotExist(err) {
		t.Error("more backups kept than log_max_backups")
	}
	cur := readLines(t, logFile)
	if cur[len(cur)-1]["user"] != "u19" {
		t.Errorf("newest line is %v, want u19 in the live file", cur[len(cur)-1]["user"])
	}
	if total >= 20 {
		t.Errorf("kept %d lines; rotation should have dropped the oldest", total)
	}
}

func TestReloadReopensRequestLog(t *testing.T) {
	// What logrotate without copytruncate needs: move the file, signal, and
	// new lines land in a fresh file at the configured path.
	logFile := filepath.Join(t.TempDir(), "requests.jsonl")
	a := usageAuth(t, UsageConfig{LogFile: logFile})
	a.Record(context.Background(), okEvent("before", "legacy", "tiny", 1))
	if err := os.Rename(logFile, logFile+".old"); err != nil {
		t.Fatal(err)
	}
	if err := a.Reload(); err != nil {
		t.Fatal(err)
	}
	a.Record(context.Background(), okEvent("after", "legacy", "tiny", 1))
	if lines := readLines(t, logFile); len(lines) != 1 || lines[0]["user"] != "after" {
		t.Errorf("new log = %v", lines)
	}
}

func TestRecorderConcurrency(t *testing.T) {
	// Run with -race. Every line must be whole and every event counted,
	// while snapshots, Totals readers and log reopening run concurrently.
	dir := t.TempDir()
	logFile := filepath.Join(dir, "requests.jsonl")
	totals := filepath.Join(dir, "totals.json")
	a, err := New(Config{Usage: UsageConfig{LogFile: logFile, TotalsFile: totals,
		LogMaxBytes: 1 << 20, SnapshotInterval: time.Millisecond}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go a.Run(ctx)

	const workers, perWorker = 8, 200
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			for i := range perWorker {
				a.Record(ctx, okEvent("u"+strconv.Itoa(w%3), core.ProtocolLegacy, "m"+strconv.Itoa(i%4), 0.5))
				if i%50 == 0 {
					_ = a.Totals()
					_ = a.Reload()
				}
			}
		})
	}
	wg.Wait()
	cancel()
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}

	var n int64
	var secs float64
	for _, u := range a.Totals() {
		n += u.Requests
		secs += u.AudioSeconds
	}
	if n != workers*perWorker || secs != workers*perWorker*0.5 {
		t.Errorf("counted %d requests / %v seconds, want %d / %v", n, secs, workers*perWorker, workers*perWorker*0.5)
	}
	if got := len(readLines(t, logFile)); got != workers*perWorker {
		t.Errorf("request log has %d whole lines, want %d", got, workers*perWorker)
	}
	var n2 int64
	for _, u := range usageAuth(t, UsageConfig{TotalsFile: totals}).Totals() {
		n2 += u.Requests
	}
	if n2 != n {
		t.Errorf("snapshot on disk has %d requests, want %d", n2, n)
	}
}

func BenchmarkRecord(b *testing.B) {
	a, err := New(Config{Usage: UsageConfig{LogFile: filepath.Join(b.TempDir(), "requests.jsonl")}}, nil)
	if err != nil {
		b.Fatal(err)
	}
	defer a.Close()
	ev := okEvent("alice", core.ProtocolLegacy, "tiny", 1)
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		a.Record(ctx, ev)
	}
}
