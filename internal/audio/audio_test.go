package audio

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// recording is a real speech recording kept in the repository for the
// legacy reference tests.
const recording = "../legacy/testdata/audio/english-front-center.wav"

// tone writes a generated 44.1 kHz stereo sine of the given length, for
// decoder paths where the content does not matter.
func tone(t *testing.T, name string, seconds float64) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	out, err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i",
		fmt.Sprintf("sine=frequency=440:sample_rate=44100:duration=%g", seconds), "-ac", "2", p).CombinedOutput()
	if err != nil {
		t.Fatalf("generate %s: %v %s", name, err, out)
	}
	return p
}

func needFFmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not in PATH")
	}
}

func wantKind(t *testing.T, err error, kind core.ErrorKind) {
	t.Helper()
	if got := core.KindOf(err); err == nil || got != kind {
		t.Fatalf("want %s, got %v (%v)", kind, got, err)
	}
}

func dirEntries(t *testing.T, dir string) []string {
	t.Helper()
	es, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range es {
		names = append(names, e.Name())
	}
	return names
}

func TestDecodeWAV(t *testing.T) {
	needFFmpeg(t)
	spool := t.TempDir()
	d := New(Config{SpoolDir: spool}, nil)
	f, err := os.Open(recording)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	pcm, err := d.DecodeFile(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	if s := pcm.Seconds(); math.Abs(s-1.428) > 0.01 {
		t.Fatalf("decoded %.3fs, want 1.428s", s)
	}
	if left := dirEntries(t, spool); len(left) != 0 {
		t.Fatalf("spool dir not cleaned: %v", left)
	}
}

// An M4A with the moov atom after mdat (ffmpeg's default without
// -movflags +faststart) decodes to nothing through a pipe. The test proves
// that first, then shows DecodeFile handles the same bytes.
func TestDecodeTrailingMoovM4A(t *testing.T) {
	needFFmpeg(t)
	src := tone(t, "long.wav", 20)
	m4a := filepath.Join(t.TempDir(), "t.m4a")
	if out, err := exec.Command("ffmpeg", "-v", "error", "-y", "-i", src, "-c:a", "aac", "-b:a", "128k", m4a).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg encode: %v %s", err, out)
	}
	data, err := os.ReadFile(m4a)
	if err != nil {
		t.Fatal(err)
	}
	if moov, mdat := bytes.Index(data, []byte("moov")), bytes.Index(data, []byte("mdat")); moov < mdat {
		t.Fatalf("test file has moov at %d before mdat at %d; need trailing moov", moov, mdat)
	}

	pipe := exec.Command("ffmpeg", "-nostdin", "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", "16000", "-")
	pipe.Stdin = bytes.NewReader(data)
	piped, _ := pipe.Output()
	if len(piped) != 0 {
		t.Fatalf("pipe decode produced %d bytes; the spool rationale no longer reproduces", len(piped))
	}

	d := New(Config{SpoolDir: t.TempDir()}, nil)
	pcm, err := d.DecodeFile(context.Background(), bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if s := pcm.Seconds(); math.Abs(s-20) > 0.1 {
		t.Fatalf("decoded %.2fs, want 20s", s)
	}
}

func TestDecodeGarbageRedactsAndDumps(t *testing.T) {
	needFFmpeg(t)
	spool, dump := t.TempDir(), filepath.Join(t.TempDir(), "failed")
	d := New(Config{SpoolDir: spool, FailedUploadsDir: dump}, nil)
	data := []byte("this is not audio at all, just text")
	_, err := d.DecodeFile(context.Background(), bytes.NewReader(data))
	wantKind(t, err, core.KindBadAudio)
	want := "Failed to decode audio (35 bytes, magic=74686973206973206e6f7420)"
	if got := core.ClientMessage(err); got != want {
		t.Fatalf("client message %q, want %q", got, want)
	}
	if !strings.Contains(err.Error(), "Invalid data") {
		t.Errorf("ffmpeg diagnostics missing from cause: %v", err)
	}
	if left := dirEntries(t, spool); len(left) != 0 {
		t.Fatalf("spool dir not cleaned: %v", left)
	}
	names := dirEntries(t, dump)
	if len(names) != 2 {
		t.Fatalf("dump dir has %v, want upload + stderr", names)
	}
	for _, n := range names {
		if strings.HasSuffix(n, ".upload") {
			got, _ := os.ReadFile(filepath.Join(dump, n))
			if !bytes.Equal(got, data) {
				t.Fatalf("dumped upload differs")
			}
		}
	}
}

func TestDecodeEmptyUpload(t *testing.T) {
	needFFmpeg(t)
	d := New(Config{SpoolDir: t.TempDir()}, nil)
	_, err := d.DecodeFile(context.Background(), bytes.NewReader(nil))
	wantKind(t, err, core.KindBadAudio)
	if got := core.ClientMessage(err); got != "Failed to decode audio (0 bytes, magic=)" {
		t.Fatalf("message %q", got)
	}
}

// A valid container holding no audio samples must be an error, not an
// empty transcript.
func TestDecodeZeroSamples(t *testing.T) {
	needFFmpeg(t)
	wav := filepath.Join(t.TempDir(), "empty.wav")
	if out, err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0", wav).CombinedOutput(); err != nil {
		t.Fatalf("%v %s", err, out)
	}
	data, _ := os.ReadFile(wav)
	d := New(Config{SpoolDir: t.TempDir()}, nil)
	_, err := d.DecodeFile(context.Background(), bytes.NewReader(data))
	wantKind(t, err, core.KindBadAudio)
	if got := core.ClientMessage(err); !strings.HasPrefix(got, "Audio decoded to 0 samples (") {
		t.Fatalf("message %q", got)
	}
}

func TestDecodeKillsFFmpegOnCancel(t *testing.T) {
	script := filepath.Join(t.TempDir(), "slow-ffmpeg")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	d := New(Config{FFmpegPath: script, SpoolDir: t.TempDir()}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := d.DecodeFile(ctx, bytes.NewReader([]byte("x")))
	wantKind(t, err, core.KindCanceled)
	if el := time.Since(start); el > 5*time.Second {
		t.Fatalf("returned after %v; ffmpeg was not killed", el)
	}
}

func TestDecodeMissingFFmpegIsInternal(t *testing.T) {
	d := New(Config{FFmpegPath: "/nonexistent/ffmpeg", SpoolDir: t.TempDir()}, nil)
	_, err := d.DecodeFile(context.Background(), bytes.NewReader([]byte("x")))
	wantKind(t, err, core.KindInternal)
	if d.Check(context.Background()) == nil {
		t.Fatal("Check accepted a missing ffmpeg")
	}
}

type failingReader struct{ err error }

func (r failingReader) Read([]byte) (int, error) { return 0, r.err }

func TestDecodeReadErrorKeepsCause(t *testing.T) {
	d := New(Config{SpoolDir: t.TempDir()}, nil)
	boom := errors.New("http: request body too large")
	_, err := d.DecodeFile(context.Background(), failingReader{boom})
	wantKind(t, err, core.KindBadAudio)
	if !errors.Is(err, boom) {
		t.Fatalf("cause lost: %v", err)
	}
}

func TestCapBuffer(t *testing.T) {
	c := &capBuffer{limit: 10}
	for i := 0; i < 100; i++ {
		_, _ = c.Write([]byte("ab"))
	}
	_, _ = c.Write([]byte("END"))
	s := c.String()
	if !strings.HasPrefix(s, "ababa") || !strings.HasSuffix(s, "abEND") || !strings.Contains(s, "dropped") {
		t.Fatalf("got %q", s)
	}
}

// --- FromRaw ---

func le16(vals ...int16) []byte {
	b := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

func samples(p core.PCM) []int16 {
	out := make([]int16, len(p)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(p[2*i:]))
	}
	return out
}

func TestFromRawFastPathIsZeroCopy(t *testing.T) {
	d := New(Config{}, nil)
	in := le16(1, 2, 3)
	out, err := d.FromRaw(context.Background(), in, 16000, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if &out[0] != &in[0] || len(out) != len(in) {
		t.Fatal("fast path copied the input")
	}
}

func TestFromRawLegacyErrorTexts(t *testing.T) {
	d := New(Config{}, nil)
	_, err := d.FromRaw(context.Background(), []byte{1, 2, 3}, 16000, 2, 1)
	wantKind(t, err, core.KindBadAudio)
	if got := core.ClientMessage(err); got != "Raw PCM payload has an odd length (3 bytes) (3 bytes, magic=010203)" {
		t.Fatalf("message %q", got)
	}
	_, err = d.FromRaw(context.Background(), nil, 16000, 2, 1)
	if got := core.ClientMessage(err); got != "Audio decoded to 0 samples (0 bytes, magic=)" {
		t.Fatalf("message %q", got)
	}
}

func TestFromRawWidthsAndDownmix(t *testing.T) {
	d := New(Config{}, nil)
	cases := []struct {
		name     string
		data     []byte
		width    int
		channels int
		want     []int16
	}{
		{"u8", []byte{128, 255, 0, 192}, 1, 1, []int16{0, 32512, -32768, 16384}},
		{"s24", []byte{0, 0, 0x40, 0, 0, 0xc0, 0xff, 0xff, 0x7f}, 3, 1, []int16{16384, -16384, 32767}},
		{"s32", []byte{0, 0, 0, 0x40, 0, 0, 0, 0x80}, 4, 1, []int16{16384, -32768}},
		{"s16 stereo", le16(1000, 3000, -2000, 0), 2, 2, []int16{2000, -1000}},
		{"u8 stereo", []byte{128, 255, 0, 128}, 1, 2, []int16{16256, -16384}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := d.FromRaw(context.Background(), tc.data, 16000, tc.width, tc.channels)
			if err != nil {
				t.Fatal(err)
			}
			got := samples(out)
			if len(got) != len(tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
			for i := range got {
				if d := int(got[i]) - int(tc.want[i]); d > 1 || d < -1 {
					t.Fatalf("got %v want %v", got, tc.want)
				}
			}
		})
	}
}

func TestFromRawRejects(t *testing.T) {
	d := New(Config{}, nil)
	for _, tc := range []struct {
		data                  []byte
		rate, width, channels int
	}{
		{le16(1, 2), 16000, 0, 1},
		{le16(1, 2), 16000, 5, 1},
		{le16(1, 2), 0, 2, 1},
		{le16(1, 2), -16000, 2, 1},
		{le16(1, 2), 16000, 2, 0},
		{le16(1, 2, 3), 16000, 2, 2}, // 6 bytes, frame 4
		{[]byte{1, 2}, 48000, 3, 1},
		{nil, 48000, 2, 1},
	} {
		_, err := d.FromRaw(context.Background(), tc.data, tc.rate, tc.width, tc.channels)
		wantKind(t, err, core.KindBadAudio)
	}
}

func sine(rate int, freq, amp, seconds float64) []byte {
	n := int(float64(rate) * seconds)
	vals := make([]int16, n)
	for i := range vals {
		vals[i] = int16(amp * 32767 * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return le16(vals...)
}

// rms over the middle half, away from edge effects.
func rms(s []int16) float64 {
	mid := s[len(s)/4 : 3*len(s)/4]
	var acc float64
	for _, v := range mid {
		f := float64(v) / 32768
		acc += f * f
	}
	return math.Sqrt(acc / float64(len(mid)))
}

func TestResamplePassbandAndAliasRejection(t *testing.T) {
	d := New(Config{}, nil)
	ideal := 0.5 / math.Sqrt2
	for _, rate := range []int{8000, 11025, 22050, 44100, 48000} {
		out, err := d.FromRaw(context.Background(), sine(rate, 1000, 0.5, 1), rate, 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		s := samples(out)
		if want := 16000; math.Abs(float64(len(s)-want)) > 1 {
			t.Fatalf("rate %d: %d samples, want %d", rate, len(s), want)
		}
		if r := rms(s); math.Abs(r-ideal)/ideal > 0.01 {
			t.Errorf("rate %d: 1 kHz tone rms %.4f, want %.4f", rate, r, ideal)
		}
	}
	// 10 kHz is above the 8 kHz output Nyquist; plain decimation or linear
	// interpolation would fold it to 6 kHz at nearly full level.
	out, err := d.FromRaw(context.Background(), sine(48000, 10000, 0.5, 1), 48000, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r := rms(samples(out)); r > ideal/1000 {
		t.Errorf("alias rms %.6f, want < -60 dB of %.4f", r, ideal)
	}
}

func TestFromRawCancel(t *testing.T) {
	d := New(Config{}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := d.FromRaw(ctx, sine(48000, 1000, 0.5, 2), 48000, 2, 1)
	wantKind(t, err, core.KindCanceled)
}

func BenchmarkResample48k(b *testing.B) {
	d := New(Config{}, nil)
	data := sine(48000, 1000, 0.5, 10)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := d.FromRaw(context.Background(), data, 48000, 2, 1); err != nil {
			b.Fatal(err)
		}
	}
}
