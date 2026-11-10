// Package audio implements core.AudioDecoder: ffmpeg for uploaded files and
// a pure Go converter for headerless PCM.
package audio

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Config configures the decoder. Zero values select the documented defaults.
type Config struct {
	// FFmpegPath is the ffmpeg binary, looked up in PATH when it has no
	// slash. Default "ffmpeg".
	FFmpegPath string `yaml:"ffmpeg_path"`
	// SpoolDir holds uploads while ffmpeg reads them. Default: os.TempDir().
	// Point it at real disk when /tmp is a small tmpfs: uploads can be large.
	SpoolDir string `yaml:"spool_dir"`
	// FailedUploadsDir, when set, keeps every upload that fails to decode
	// there together with the ffmpeg output, for offline replay. Off by
	// default so production does not archive user audio.
	FailedUploadsDir string `yaml:"failed_uploads_dir"`
}

// Decoder implements core.AudioDecoder.
type Decoder struct {
	cfg Config
	log *slog.Logger
}

var _ core.AudioDecoder = (*Decoder)(nil)

// New returns a decoder. A nil log uses slog.Default(). It does not check
// for ffmpeg; call Check at startup for that.
func New(cfg Config, log *slog.Logger) *Decoder {
	if log == nil {
		log = slog.Default()
	}
	return &Decoder{cfg: cfg.Resolved(), log: log}
}

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it.
func (c Config) Resolved() Config {
	if c.FFmpegPath == "" {
		c.FFmpegPath = "ffmpeg"
	}
	if c.SpoolDir == "" {
		c.SpoolDir = os.TempDir()
	}
	return c
}

// Check verifies that ffmpeg runs and the spool directory is writable.
func (d *Decoder) Check(ctx context.Context) error {
	out, err := exec.CommandContext(ctx, d.cfg.FFmpegPath, "-hide_banner", "-version").Output()
	if err != nil {
		return fmt.Errorf("audio: ffmpeg %q not usable: %w", d.cfg.FFmpegPath, err)
	}
	f, err := os.CreateTemp(d.cfg.SpoolDir, "whisper-hub-check-*")
	if err != nil {
		return fmt.Errorf("audio: spool dir not writable: %w", err)
	}
	_ = f.Close()
	_ = os.Remove(f.Name())
	first, _, _ := strings.Cut(string(out), "\n")
	d.log.Debug("ffmpeg found", "path", d.cfg.FFmpegPath, "version", first)
	return nil
}

const magicLen = 12

// headCapture keeps the first magicLen bytes written to it.
type headCapture struct{ b []byte }

func (h *headCapture) Write(p []byte) (int, error) {
	if n := magicLen - len(h.b); n > 0 {
		h.b = append(h.b, p[:min(n, len(p))]...)
	}
	return len(p), nil
}

// clientDetail formats an error the way upstream whisper-asr-webservice
// does, so legacy clients see the same 400 detail text.
func clientDetail(reason string, size int64, magic []byte) string {
	return fmt.Sprintf("%s (%d bytes, magic=%s)", reason, size, hex.EncodeToString(magic))
}

// DecodeFile spools r to a temp file and decodes it with ffmpeg to mono
// 16 kHz s16le.
//
// The spool file is the point: MP4/M4A with the moov atom at the end (as
// Android MediaMuxer writes them) needs seekable input, and on a pipe
// ffmpeg silently decodes such a file to zero samples.
func (d *Decoder) DecodeFile(ctx context.Context, r io.Reader) (core.PCM, error) {
	started := time.Now()
	f, err := os.CreateTemp(d.cfg.SpoolDir, "whisper-hub-*.upload")
	if err != nil {
		return nil, core.WrapError(core.KindInternal, err, "cannot store upload")
	}
	spool := f.Name()
	keep := false
	defer func() {
		if keep {
			return
		}
		if err := os.Remove(spool); err != nil {
			// A recurring failure here means the spool dir is filling up.
			d.log.Warn("cannot remove spooled upload", "path", spool, "err", err)
		}
	}()

	var head headCapture
	size, err := io.Copy(io.MultiWriter(f, &head), r)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		if ctx.Err() != nil {
			return nil, core.WrapError(core.KindCanceled, err, "request canceled")
		}
		d.log.Warn("cannot read upload", "bytes", size, "err", err)
		return nil, core.WrapError(core.KindBadAudio, err, "%s", clientDetail("Failed to read upload", size, head.b))
	}
	d.log.Debug("decoding upload", "bytes", size, "magic", hex.EncodeToString(head.b), "spool", spool)

	var stdout bytes.Buffer
	stderr := &capBuffer{limit: 64 << 10}
	args := []string{"-nostdin", "-hide_banner", "-threads", "0", "-i", spool,
		"-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", strconv.Itoa(core.SampleRate), "-"}
	cmd := exec.CommandContext(ctx, d.cfg.FFmpegPath, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = 5 * time.Second
	d.log.Debug("running ffmpeg", "args", args)
	runErr := cmd.Run()
	if ctx.Err() != nil {
		return nil, core.WrapError(core.KindCanceled, ctx.Err(), "request canceled")
	}
	var reason string
	switch {
	case runErr != nil:
		var exitErr *exec.ExitError
		if !errors.As(runErr, &exitErr) {
			// ffmpeg missing or not executable: our fault, not the client's.
			return nil, core.WrapError(core.KindInternal, runErr, "audio decoder unavailable")
		}
		reason = "Failed to decode audio"
	case stdout.Len()%2 != 0:
		reason = fmt.Sprintf("Raw PCM payload has an odd length (%d bytes)", stdout.Len())
	case stdout.Len() == 0:
		reason = "Audio decoded to 0 samples"
	}
	if reason != "" {
		keep = d.decodeFailed(reason, spool, size, head.b, stderr.String())
		detail := clientDetail(reason, size, head.b)
		cause := runErr
		if cause == nil {
			cause = errors.New(reason)
		}
		return nil, core.WrapError(core.KindBadAudio, fmt.Errorf("%w: %s", cause, excerpt(stderr.String(), 1500)), "%s", detail)
	}
	pcm := core.PCM(stdout.Bytes())
	d.log.Debug("decoded upload", "bytes", size, "samples", len(pcm)/2, "seconds", pcm.Seconds(),
		"elapsed_ms", time.Since(started).Milliseconds())
	return pcm, nil
}

// decodeFailed logs full diagnostics and optionally keeps the upload. It
// reports whether the spool file was moved away.
func (d *Decoder) decodeFailed(reason, spool string, size int64, magic []byte, stderr string) (moved bool) {
	dump := ""
	if d.cfg.FailedUploadsDir != "" {
		var err error
		dump, err = dumpUpload(d.cfg.FailedUploadsDir, spool, stderr)
		if err != nil {
			d.log.Error("cannot dump failing upload", "dir", d.cfg.FailedUploadsDir, "err", err)
			dump = ""
		}
	}
	if strings.TrimSpace(stderr) == "" {
		stderr = "(none)"
	}
	d.log.Warn(reason, "bytes", size, "magic", hex.EncodeToString(magic), "dump", orDefault(dump, "disabled"),
		"ffmpeg_output", strings.TrimSpace(stderr))
	return dump != ""
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func dumpUpload(dir, spool, stderr string) (string, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	now := time.Now()
	base := filepath.Join(dir, fmt.Sprintf("failed-%s-%09d", now.Format("20060102-150405"), now.Nanosecond()))
	path := base + ".upload"
	if err := os.Rename(spool, path); err != nil {
		// Different filesystem: fall back to a copy.
		if err := copyFile(spool, path); err != nil {
			return "", err
		}
		_ = os.Remove(spool)
	}
	if err := os.WriteFile(base+".stderr.txt", []byte(stderr), 0o600); err != nil {
		return path, err
	}
	return path, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}
	return out.Close()
}

// capBuffer keeps the head and the tail of a possibly huge ffmpeg stderr:
// the input analysis is at the start, the fatal error at the end.
type capBuffer struct {
	limit   int
	head    []byte
	tail    []byte
	dropped int64
}

func (c *capBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if room := c.limit/2 - len(c.head); room > 0 {
		k := min(room, len(p))
		c.head = append(c.head, p[:k]...)
		p = p[k:]
	}
	if len(p) > 0 {
		c.tail = append(c.tail, p...)
		if over := len(c.tail) - c.limit/2; over > 0 {
			c.dropped += int64(over)
			c.tail = append(c.tail[:0], c.tail[over:]...)
		}
	}
	return n, nil
}

func (c *capBuffer) String() string {
	if c.dropped == 0 {
		return string(c.head) + string(c.tail)
	}
	return fmt.Sprintf("%s\n[... %d bytes dropped ...]\n%s", c.head, c.dropped, c.tail)
}

func excerpt(s string, limit int) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "(no ffmpeg diagnostics)"
	}
	if len(s) <= limit {
		return s
	}
	return s[:limit/2] + "\n[...]\n" + s[len(s)-limit/2:]
}
