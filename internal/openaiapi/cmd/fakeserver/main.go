// Command fakeserver runs the OpenAI frontend against a canned core.Service
// so real OpenAI clients (SDKs, curl) can be checked without engine workers.
//
// The model name selects a failure: missing, forbidden, unauth, busy, boom
// (internal) and midfail (fails after the first streamed segment). An upload
// starting with "corrupt" fails decoding. Any other model succeeds.
package main

import (
	"bytes"
	"context"
	"flag"
	"io"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/openaiapi"
)

type decoder struct{}

func (decoder) DecodeFile(ctx context.Context, r io.Reader) (core.PCM, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if bytes.HasPrefix(b, []byte("corrupt")) {
		return nil, core.Errorf(core.KindBadAudio, "Audio file might be corrupted or unsupported")
	}
	// 3.5 s of silence regardless of input: the fake engine ignores audio.
	return make(core.PCM, core.SampleRate*2*35/10), nil
}

func (decoder) FromRaw(ctx context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	return core.PCM(data), nil
}

type service struct{ delay time.Duration }

var segments = []core.Segment{
	{ID: 0, Seek: 0, Start: 0, End: 1.2, Text: " Hello world.", Tokens: []int{50364, 2425, 1002, 13},
		AvgLogprob: -0.21, CompressionRatio: 0.8, NoSpeechProb: 0.01,
		Words: []core.Word{{Word: " Hello", Start: 0, End: 0.5, Probability: 0.9}, {Word: " world.", Start: 0.5, End: 1.2, Probability: 0.8}}},
	{ID: 1, Seek: 0, Start: 1.2, End: 2.6, Text: " This is a test.", Tokens: []int{639, 307, 257, 1500, 13},
		AvgLogprob: -0.3, CompressionRatio: 0.9, NoSpeechProb: 0.02,
		Words: []core.Word{{Word: " This", Start: 1.2, End: 1.5, Probability: 0.9}, {Word: " is", Start: 1.5, End: 1.7, Probability: 0.9},
			{Word: " a", Start: 1.7, End: 1.8, Probability: 0.9}, {Word: " test.", Start: 1.8, End: 2.6, Probability: 0.9}}},
	{ID: 2, Seek: 0, Start: 2.6, End: 3.5, Text: " Goodbye.", Tokens: []int{15528, 13},
		AvgLogprob: -0.25, CompressionRatio: 0.7, NoSpeechProb: 0.03,
		Words: []core.Word{{Word: " Goodbye.", Start: 2.6, End: 3.5, Probability: 0.95}}},
}

func (s *service) Transcribe(ctx context.Context, req *core.Request, onSegment func(core.Segment) error) (*core.Result, error) {
	slog.Debug("fake: transcribe", "model", req.Model, "task", req.Task, "language", req.Language,
		"words", req.WordTimestamps, "temperature", req.Temperature, "prompt", req.Prompt)
	switch req.Model {
	case "missing":
		return nil, core.Errorf(core.KindModelNotFound, "The model `missing` does not exist or you do not have access to it.")
	case "forbidden":
		return nil, core.Errorf(core.KindForbidden, "You are not allowed to use model `forbidden`.")
	case "unauth":
		return nil, core.Errorf(core.KindUnauthenticated, "Incorrect API key provided.")
	case "busy":
		return nil, core.Errorf(core.KindUnavailable, "No engine worker is available, try again later.")
	case "boom":
		return nil, core.WrapError(core.KindInternal, io.ErrUnexpectedEOF, "engine crashed")
	}
	res := &core.Result{Model: "tiny", Engine: "fake", Language: "en", LanguageProbability: 0.99,
		Duration: req.Audio.Seconds()}
	if req.Language != "" {
		res.Language = req.Language
	}
	for i, seg := range segments {
		if !req.WordTimestamps {
			seg.Words = nil
		}
		if onSegment != nil {
			if i == 1 && req.Model == "midfail" {
				return nil, core.Errorf(core.KindUnavailable, "engine worker went away")
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(s.delay):
			}
			if err := onSegment(seg); err != nil {
				return nil, err
			}
		}
		res.Segments = append(res.Segments, seg)
		res.Text += seg.Text
	}
	return res, nil
}

func (s *service) DetectLanguage(ctx context.Context, req *core.DetectRequest) (*core.Detection, error) {
	return &core.Detection{Language: "en", Confidence: 0.99, Engine: "fake", Model: "tiny"}, nil
}

func (s *service) Models(ctx context.Context) []core.Model {
	return []core.Model{{ID: "tiny", Engine: "fake", Default: true, Aliases: []string{"whisper-1"}}, {ID: "base", Engine: "fake"}}
}

func main() {
	addr := flag.String("addr", ":9000", "listen address")
	maxUpload := flag.Int64("max-upload", 0, "max upload bytes, 0 = unlimited")
	delay := flag.Duration("delay", 200*time.Millisecond, "delay before each streamed segment")
	flag.Parse()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))

	mux := http.NewServeMux()
	openaiapi.New(openaiapi.Config{MaxUploadBytes: *maxUpload}, &service{delay: *delay}, decoder{}, nil).Register(mux)
	slog.Info("fakeserver listening", "addr", *addr, "max_upload", *maxUpload)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		slog.Error("fakeserver stopped", "error", err)
		os.Exit(1)
	}
}
