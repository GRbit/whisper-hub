// Command fakeserver runs the Wyoming server on an in-memory Service, so
// real Wyoming clients can be checked against this package without engines.
//
// The transcript is "heard N.NN seconds" (the converted audio length), sent
// as three segments, so a client can verify that its audio arrived whole.
package main

import (
	"context"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/wyoming"
)

func main() {
	listen := flag.String("listen", ":10300", "listen address")
	stream := flag.Bool("stream", false, "advertise and send transcript streaming events")
	zc := flag.Bool("zeroconf", false, "announce over mDNS")
	zcName := flag.String("zeroconf-name", "", "mDNS instance name")
	zcHost := flag.String("zeroconf-host", "", "IPv4 address to advertise")
	fail := flag.String("fail", "", "fail every transcription with this core error kind name (e.g. forbidden)")
	delay := flag.Duration("segment-delay", 20*time.Millisecond, "delay before each segment")
	debug := flag.Bool("debug", false, "debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	cfg := wyoming.Config{
		Listen:            *listen,
		StreamTranscripts: *stream,
		Version:           "fake",
		Zeroconf:          wyoming.ZeroconfConfig{Enabled: *zc, Name: *zcName, Host: *zcHost},
	}
	srv, err := wyoming.New(cfg, &service{fail: *fail, delay: *delay}, decoder{}, nil, log)
	if err != nil {
		log.Error("config", "err", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := srv.ListenAndServe(ctx); err != nil {
		log.Error("serve", "err", err)
		os.Exit(1)
	}
}

type service struct {
	fail  string
	delay time.Duration
}

func (s *service) Models(context.Context) []core.Model {
	return []core.Model{
		{ID: "tiny", Aliases: []string{"whisper-1"}, Engine: "faster_whisper", Languages: core.Languages(), Default: true},
		{ID: "base.en", Engine: "faster_whisper", Languages: []string{"en"}},
	}
}

func (s *service) DetectLanguage(context.Context, *core.DetectRequest) (*core.Detection, error) {
	return nil, core.Errorf(core.KindBadRequest, "not supported")
}

func (s *service) Transcribe(ctx context.Context, req *core.Request, onSegment func(core.Segment) error) (*core.Result, error) {
	if s.fail != "" {
		for k := core.KindInternal; k <= core.KindCanceled; k++ {
			if k.String() == s.fail {
				return nil, core.Errorf(k, "fake failure")
			}
		}
		return nil, errors.New("fake failure")
	}
	words := []string{" heard", fmt.Sprintf(" %.2f", req.Audio.Seconds()), " seconds"}
	var text strings.Builder
	for i, w := range words {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return nil, core.WrapError(core.KindCanceled, ctx.Err(), "canceled")
		}
		text.WriteString(w)
		if onSegment != nil {
			if err := onSegment(core.Segment{ID: i, Text: w}); err != nil {
				return nil, err
			}
		}
	}
	lang := req.Language
	if lang == "" {
		lang = "en"
	}
	return &core.Result{Model: "tiny", Engine: "fake", Language: lang, Duration: req.Audio.Seconds(), Text: text.String()}, nil
}

// decoder downmixes and resamples with nearest-neighbour picking: crude, but
// the length is exact, which is all the interop check looks at.
type decoder struct{}

func (decoder) DecodeFile(context.Context, io.Reader) (core.PCM, error) {
	return nil, core.Errorf(core.KindBadAudio, "not supported")
}

func (decoder) FromRaw(_ context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	frame := width * channels
	frames := len(data) / frame
	outFrames := int(int64(frames) * core.SampleRate / int64(rate))
	out := make(core.PCM, outFrames*2)
	for i := range outFrames {
		src := int(int64(i) * int64(rate) / core.SampleRate)
		var sum int64
		for ch := range channels {
			off := src*frame + ch*width
			sum += int64(sample(data[off:off+width])) >> (8 * (width - 2) * boolInt(width > 2))
		}
		v := sum / int64(channels)
		if width == 1 {
			v <<= 8
		}
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(v)))
	}
	return out, nil
}

func sample(b []byte) int32 {
	switch len(b) {
	case 1:
		return int32(int8(b[0]))
	case 2:
		return int32(int16(binary.LittleEndian.Uint16(b)))
	case 3:
		return int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24) >> 8
	default:
		return int32(binary.LittleEndian.Uint32(b))
	}
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
