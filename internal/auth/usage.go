package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Counter is a usage breakdown bucket (one protocol or one model).
type Counter struct {
	Requests       int64   `json:"requests"`
	OKRequests     int64   `json:"ok_requests"`
	FailedRequests int64   `json:"failed_requests"`
	AudioSeconds   float64 `json:"audio_seconds"`
}

// UserTotals is the running total for one user.
//
// The top-level fields are those of the old sidecar usage.json, so its
// snapshot loads unchanged. Billable and wasted audio stay separate: folding
// them together would either overcharge for failed work or hide how much
// processing time is being lost.
type UserTotals struct {
	Requests       int64 `json:"requests"`
	OKRequests     int64 `json:"ok_requests"`
	FailedRequests int64 `json:"failed_requests"`
	// AudioSeconds counts successfully transcribed audio only.
	AudioSeconds float64 `json:"audio_seconds"`
	// FailedSeconds is audio that decoded but whose job then failed.
	FailedSeconds float64 `json:"failed_seconds"`
	// UnknownRequests failed before the audio length was known.
	UnknownRequests int64 `json:"unknown_requests"`
	// BytesIn is carried over from sidecar snapshots; the hub does not
	// see upload sizes, so it never grows.
	BytesIn int64 `json:"bytes_in,omitempty"`
	// ProcessingSeconds is engine time (elapsed minus queued) over all jobs.
	ProcessingSeconds float64 `json:"processing_seconds"`

	FirstSeen time.Time `json:"first_seen"`
	LastSeen  time.Time `json:"last_seen"`

	ByProtocol map[string]Counter `json:"by_protocol,omitempty"`
	ByModel    map[string]Counter `json:"by_model,omitempty"`
}

func (t UserTotals) clone() UserTotals {
	t.ByProtocol = maps.Clone(t.ByProtocol)
	t.ByModel = maps.Clone(t.ByModel)
	return t
}

// logEntry is a request log line. Separate from core.UsageEvent so the
// on-disk field names are a deliberate contract.
type logEntry struct {
	TS            string  `json:"ts"`
	User          string  `json:"user"`
	AttemptedUser string  `json:"attempted_user,omitempty"`
	Protocol      string  `json:"protocol"`
	Endpoint      string  `json:"endpoint"`
	Model         string  `json:"model,omitempty"`
	Engine        string  `json:"engine,omitempty"`
	Task          string  `json:"task,omitempty"`
	AudioSeconds  float64 `json:"audio_seconds"`
	QueuedMS      int64   `json:"queued_ms"`
	ElapsedMS     int64   `json:"elapsed_ms"`
	Status        string  `json:"status"`
	Error         string  `json:"error,omitempty"`
	ClientIP      string  `json:"client_ip,omitempty"`
	Reason        string  `json:"reason,omitempty"`
}

// recorder keeps per-user totals, persists them, and appends the request
// log. It implements core.UsageRecorder.
type recorder struct {
	cfg UsageConfig
	// totalsPath is where totals persist: TotalsFile, else
	// DefaultTotalsFile when usable, else "" (memory only).
	totalsPath string

	// Guards totals only; the log has its own lock so a slow disk write
	// never blocks accounting.
	mu     sync.Mutex
	totals map[string]*UserTotals
	dirty  bool

	logMu   sync.Mutex
	log     *os.File
	logSize int64

	saveMu sync.Mutex

	now func() time.Time
}

var _ core.UsageRecorder = (*recorder)(nil)

// newRecorder opens the request log and restores the previous totals, so
// counters survive a restart.
func newRecorder(cfg UsageConfig) (*recorder, error) {
	cfg = cfg.resolved()
	r := &recorder{cfg: cfg, totals: make(map[string]*UserTotals), now: time.Now}
	switch {
	case cfg.TotalsFile != "":
		r.totalsPath = cfg.TotalsFile
		if err := os.MkdirAll(filepath.Dir(cfg.TotalsFile), 0o755); err != nil {
			return nil, fmt.Errorf("create usage totals dir: %w", err)
		}
		if cfg.DefaultTotalsFile != "" && r.moveFrom(cfg.DefaultTotalsFile) {
			break
		}
		r.restoreOrReset()
	case cfg.DefaultTotalsFile != "":
		// The default was not asked for, so a read-only /data or a non-root
		// user must not stop the service: degrade to memory instead.
		if err := checkWritableDir(filepath.Dir(cfg.DefaultTotalsFile)); err != nil {
			slog.Warn("auth: default usage totals file is not writable, keeping totals in memory only",
				"path", cfg.DefaultTotalsFile, "error", err)
			break
		}
		r.totalsPath = cfg.DefaultTotalsFile
		r.restoreOrReset()
	}
	slog.Debug("auth: usage totals location", "path", r.totalsPath)
	if err := r.reopenLog(); err != nil {
		return nil, err
	}
	return r, nil
}

func checkWritableDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".write-check-*")
	if err != nil {
		return err
	}
	f.Close()
	return os.Remove(f.Name())
}

func (r *recorder) restoreOrReset() {
	if err := r.restore(); err != nil {
		// Not fatal: the request log is authoritative and totals can be
		// rebuilt from it.
		slog.Error("auth: could not restore usage totals, starting from zero",
			"path", r.totalsPath, "error", err)
	}
}

// moveFrom moves the totals in old to totalsPath while totalsPath is
// missing or empty, and reports whether it loaded them. The old file is
// removed only after the new one is written, so a failure at any step
// leaves the data in at least one of the two.
func (r *recorder) moveFrom(old string) bool {
	if sameFile(old, r.totalsPath) {
		return false
	}
	if st, err := os.Stat(r.totalsPath); err == nil && st.Size() > 0 {
		slog.Debug("auth: usage totals file has content, default file left alone",
			"path", r.totalsPath, "default", old)
		return false
	}
	raw, err := os.ReadFile(old)
	if errors.Is(err, fs.ErrNotExist) || (err == nil && len(raw) == 0) {
		return false
	}
	users := 0
	if err == nil {
		var totals map[string]*UserTotals
		if totals, err = parseTotals(raw); err == nil {
			users = len(totals)
			r.mu.Lock()
			r.totals = totals
			r.mu.Unlock()
		}
	}
	if err != nil {
		slog.Warn("auth: could not read the default usage totals file, not moving it",
			"from", old, "to", r.totalsPath, "error", err)
		return false
	}
	if err := writeAtomic(r.totalsPath, raw); err != nil {
		// The totals are loaded; the next snapshot retries the write.
		r.mu.Lock()
		r.dirty = true
		r.mu.Unlock()
		slog.Warn("auth: could not move usage totals, the old file is kept",
			"from", old, "to", r.totalsPath, "error", err)
		return true
	}
	if err := os.Remove(old); err != nil {
		slog.Warn("auth: usage totals moved, but the old file could not be removed; it is no longer read",
			"from", old, "to", r.totalsPath, "error", err)
		return true
	}
	slog.Info("auth: usage totals moved", "from", old, "to", r.totalsPath, "users", users)
	return true
}

// sameFile also catches one file reached through two spellings or a bind
// mount.
func sameFile(a, b string) bool {
	if filepath.Clean(a) == filepath.Clean(b) {
		return true
	}
	sa, errA := os.Stat(a)
	sb, errB := os.Stat(b)
	return errA == nil && errB == nil && os.SameFile(sa, sb)
}

func parseTotals(raw []byte) (map[string]*UserTotals, error) {
	totals := make(map[string]*UserTotals)
	if err := json.Unmarshal(raw, &totals); err != nil {
		return nil, err
	}
	for name, t := range totals {
		if t == nil {
			delete(totals, name)
		}
	}
	return totals, nil
}

func (r *recorder) restore() error {
	raw, err := os.ReadFile(r.totalsPath)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	totals, err := parseTotals(raw)
	if err != nil {
		return err
	}
	r.mu.Lock()
	r.totals = totals
	r.mu.Unlock()
	slog.Info("auth: usage totals restored", "path", r.totalsPath, "users", len(totals))
	return nil
}

func (r *recorder) reopenLog() error {
	if r.cfg.LogFile == "" {
		return nil
	}
	r.logMu.Lock()
	defer r.logMu.Unlock()
	if r.log != nil {
		_ = r.log.Close()
		r.log = nil
	}
	if err := os.MkdirAll(filepath.Dir(r.cfg.LogFile), 0o755); err != nil {
		return fmt.Errorf("create request log dir: %w", err)
	}
	return r.openLogLocked()
}

func (r *recorder) openLogLocked() error {
	f, err := os.OpenFile(r.cfg.LogFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o640)
	if err != nil {
		return fmt.Errorf("open request log: %w", err)
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return fmt.Errorf("stat request log: %w", err)
	}
	r.log, r.logSize = f, st.Size()
	return nil
}

// Record files one finished job into the totals and the request log, and
// writes the per-job Info log line.
//
// Totals first: tests use the log line as a barrier and rely on the totals
// already being in place once it appears.
func (r *recorder) Record(_ context.Context, ev core.UsageEvent) {
	if ev.Time.IsZero() {
		ev.Time = r.now()
	}
	if ev.User == "" {
		ev.User = "anonymous"
	}
	r.updateTotals(ev)
	r.appendLog(logEntry{
		TS:           ev.Time.UTC().Format(time.RFC3339Nano),
		User:         ev.User,
		Protocol:     ev.Protocol,
		Endpoint:     ev.Endpoint,
		Model:        ev.Model,
		Engine:       ev.Engine,
		Task:         string(ev.Task),
		AudioSeconds: ev.AudioSeconds,
		QueuedMS:     ev.Queued.Milliseconds(),
		ElapsedMS:    ev.Elapsed.Milliseconds(),
		Status:       ev.Status,
		Error:        ev.Error,
	})
	attrs := []any{"user", ev.User, "protocol", ev.Protocol, "endpoint", ev.Endpoint,
		"model", ev.Model, "engine", ev.Engine, "task", string(ev.Task), "status", ev.Status,
		"audio_seconds", ev.AudioSeconds, "queued_ms", ev.Queued.Milliseconds(),
		"elapsed_ms", ev.Elapsed.Milliseconds()}
	if ev.Error != "" {
		attrs = append(attrs, "error", ev.Error)
	}
	slog.Info("job finished", attrs...)
}

func bump(m map[string]Counter, key string, ok bool, seconds float64) map[string]Counter {
	if key == "" {
		key = "unknown"
	}
	if m == nil {
		m = make(map[string]Counter)
	}
	c := m[key]
	c.Requests++
	if ok {
		c.OKRequests++
		c.AudioSeconds += seconds
	} else {
		c.FailedRequests++
	}
	m[key] = c
	return m
}

func (r *recorder) updateTotals(ev core.UsageEvent) {
	ok := ev.Status == "ok"
	r.mu.Lock()
	defer r.mu.Unlock()
	t := r.totals[ev.User]
	if t == nil {
		t = &UserTotals{FirstSeen: ev.Time}
		r.totals[ev.User] = t
	}
	t.Requests++
	t.LastSeen = ev.Time
	if proc := (ev.Elapsed - ev.Queued).Seconds(); proc > 0 {
		t.ProcessingSeconds += proc
	}
	switch {
	case ok:
		t.OKRequests++
		t.AudioSeconds += ev.AudioSeconds
	case ev.AudioSeconds > 0:
		t.FailedRequests++
		t.FailedSeconds += ev.AudioSeconds
	default:
		t.FailedRequests++
		t.UnknownRequests++
	}
	t.ByProtocol = bump(t.ByProtocol, ev.Protocol, ok, ev.AudioSeconds)
	t.ByModel = bump(t.ByModel, ev.Model, ok, ev.AudioSeconds)
	r.dirty = true
}

// recordAuthFailure logs a rejected login. Deliberately kept out of the
// totals so an attacker cannot create entries for chosen names, and carries
// no password nor any derivative of one: this log gets shipped around.
func (r *recorder) recordAuthFailure(attempted, protocol, endpoint, clientIP, reason string) {
	r.appendLog(logEntry{
		TS:            r.now().UTC().Format(time.RFC3339Nano),
		AttemptedUser: attempted,
		Protocol:      protocol,
		Endpoint:      endpoint,
		Status:        core.KindUnauthenticated.String(),
		ClientIP:      clientIP,
		Reason:        reason,
	})
}

func (r *recorder) appendLog(e logEntry) {
	if r.cfg.LogFile == "" {
		return
	}
	line, err := json.Marshal(e)
	if err != nil {
		slog.Error("auth: could not encode request log entry", "error", err)
		return
	}
	line = append(line, '\n')

	// One Write per line under the lock: concurrent jobs must not interleave
	// halves of two JSON objects onto one line.
	r.logMu.Lock()
	defer r.logMu.Unlock()
	if r.log == nil {
		return // closed
	}
	if r.cfg.LogMaxBytes > 0 && r.logSize > 0 && r.logSize+int64(len(line)) > r.cfg.LogMaxBytes {
		if err := r.rotateLocked(); err != nil {
			slog.Error("auth: request log rotation failed, appending to the current file", "error", err)
		}
	}
	if r.log == nil {
		return
	}
	n, err := r.log.Write(line)
	r.logSize += int64(n)
	if err != nil {
		slog.Error("auth: could not append to the request log", "path", r.cfg.LogFile, "error", err)
	}
}

// rotateLocked shifts path.N-1 -> path.N ... path -> path.1 and reopens.
// rename(2) replaces the oldest backup, so no explicit delete is needed.
func (r *recorder) rotateLocked() error {
	if err := r.log.Close(); err != nil {
		slog.Warn("auth: closing request log before rotation", "error", err)
	}
	r.log = nil
	base := r.cfg.LogFile
	var rerr error
	for i := r.cfg.LogMaxBackups - 1; i >= 1; i-- {
		err := os.Rename(base+"."+strconv.Itoa(i), base+"."+strconv.Itoa(i+1))
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			rerr = err
		}
	}
	if err := os.Rename(base, base+".1"); err != nil && !errors.Is(err, fs.ErrNotExist) {
		rerr = err
	}
	if err := r.openLogLocked(); err != nil {
		return errors.Join(rerr, err)
	}
	slog.Debug("auth: request log rotated", "path", base, "backups", r.cfg.LogMaxBackups)
	return rerr
}

// Totals returns a copy of the current per-user totals.
func (r *recorder) Totals() map[string]UserTotals {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]UserTotals, len(r.totals))
	for name, t := range r.totals {
		out[name] = t.clone()
	}
	return out
}

// Save writes the totals to their file if they changed since the last save.
//
// Atomic: a reader sees either the whole previous snapshot or the whole new
// one, never a half-written file.
func (r *recorder) Save() error {
	if r.totalsPath == "" {
		return nil
	}
	r.saveMu.Lock()
	defer r.saveMu.Unlock()

	r.mu.Lock()
	if !r.dirty {
		r.mu.Unlock()
		return nil
	}
	r.dirty = false
	r.mu.Unlock()
	// A Record landing between clearing dirty and taking this copy is in the
	// copy and re-marks dirty, so at worst it is written twice, never lost.
	data, err := json.MarshalIndent(r.Totals(), "", "  ")
	if err == nil {
		err = writeAtomic(r.totalsPath, append(data, '\n'))
	}
	if err != nil {
		r.mu.Lock()
		r.dirty = true
		r.mu.Unlock()
		return err
	}
	slog.Debug("auth: usage totals saved", "path", r.totalsPath)
	return nil
}

func writeAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename succeeded

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	// So the rename itself survives a power loss, not just the contents.
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		d.Close()
	}
	return nil
}

// Run saves the totals every SnapshotInterval until ctx is cancelled.
func (r *recorder) Run(ctx context.Context) {
	if r.totalsPath == "" {
		return
	}
	t := time.NewTicker(r.cfg.SnapshotInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := r.Save(); err != nil {
				slog.Error("auth: periodic usage snapshot failed", "path", r.totalsPath, "error", err)
			}
		}
	}
}

// Close persists a final snapshot and closes the request log. This is what
// makes the totals exact at shutdown rather than as of the last snapshot.
func (r *recorder) Close() error {
	err := r.Save()
	r.logMu.Lock()
	defer r.logMu.Unlock()
	if r.log != nil {
		if cerr := r.log.Close(); err == nil {
			err = cerr
		}
		r.log = nil
	}
	return err
}
