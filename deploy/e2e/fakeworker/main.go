// Command fakeworker serves the Worker API with canned transcripts, so the
// real gateway binary can be checked with real clients (openai SDK,
// wyoming package, Home Assistant replay) without a model.
//
// -script openai reproduces the canned service of
// internal/openaiapi/cmd/fakeserver: three fixed segments, plus failure
// models busy (503), boom (500) and midfail (stream error after the first
// segment). -script heard reproduces internal/wyoming/cmd/fakeserver: the
// transcript is "heard N.NN seconds" for the audio length received.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/hub/workertest"
)

var openaiSegments = []core.Segment{
	{ID: 0, Start: 0, End: 1.2, Text: " Hello world.", Tokens: []int{50364, 2425, 1002, 13},
		AvgLogprob: -0.21, CompressionRatio: 0.8, NoSpeechProb: 0.01,
		Words: []core.Word{{Word: " Hello", Start: 0, End: 0.5, Probability: 0.9}, {Word: " world.", Start: 0.5, End: 1.2, Probability: 0.8}}},
	{ID: 1, Start: 1.2, End: 2.6, Text: " This is a test.", Tokens: []int{639, 307, 257, 1500, 13},
		AvgLogprob: -0.3, CompressionRatio: 0.9, NoSpeechProb: 0.02,
		Words: []core.Word{{Word: " This", Start: 1.2, End: 1.5, Probability: 0.9}, {Word: " is", Start: 1.5, End: 1.7, Probability: 0.9},
			{Word: " a", Start: 1.7, End: 1.8, Probability: 0.9}, {Word: " test.", Start: 1.8, End: 2.6, Probability: 0.9}}},
	{ID: 2, Start: 2.6, End: 3.5, Text: " Goodbye.", Tokens: []int{15528, 13},
		AvgLogprob: -0.25, CompressionRatio: 0.7, NoSpeechProb: 0.03,
		Words: []core.Word{{Word: " Goodbye.", Start: 2.6, End: 3.5, Probability: 0.95}}},
}

func lang(job *workertest.Job) string {
	if job.Options.Language != nil {
		return *job.Options.Language
	}
	return "en"
}

func openaiScript(job *workertest.Job, s *workertest.Stream) {
	switch job.Model {
	case "busy":
		s.HTTPError("unavailable", "No engine worker is available, try again later.")
		return
	case "boom":
		s.HTTPError("internal", "engine crashed")
		return
	}
	s.Info(job.Model, job.Engine, lang(job), float64(len(job.Audio)/2)/core.SampleRate)
	text := ""
	for i, seg := range openaiSegments {
		if i == 1 && job.Model == "midfail" {
			s.Error("unavailable", "engine worker went away")
			return
		}
		if !job.Options.WordTimestamps {
			seg.Words = nil
		}
		s.Segment(seg)
		text += seg.Text
	}
	s.Done(text, lang(job), nil)
}

func heardScript(job *workertest.Job, s *workertest.Stream) {
	secs := float64(len(job.Audio)/2) / core.SampleRate
	s.Info(job.Model, job.Engine, lang(job), secs)
	words := []string{" heard", fmt.Sprintf(" %.2f", secs), " seconds"}
	text := ""
	for i, w := range words {
		s.Segment(core.Segment{ID: i, Text: w})
		text += w
	}
	s.Done(text, lang(job), nil)
}

func main() {
	listen := flag.String("listen", ":9100", "listen address")
	script := flag.String("script", "openai", "openai or heard")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	info := workertest.DefaultInfo()
	var run workertest.TranscribeFunc
	switch *script {
	case "openai":
		info.Models = []workertest.ModelInfo{{ID: "tiny", Default: true, Loaded: true}, {ID: "base", Loaded: true},
			{ID: "busy"}, {ID: "boom"}, {ID: "midfail"}, {ID: "forbidden"}}
		run = openaiScript
	case "heard":
		info.Models = []workertest.ModelInfo{{ID: "tiny", Default: true, Loaded: true}, {ID: "base.en", Loaded: true}}
		run = heardScript
	default:
		log.Error("unknown -script", "script", *script)
		os.Exit(2)
	}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Error("listen", "err", err)
		os.Exit(1)
	}
	w := workertest.NewUnstarted(info)
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		log.Debug("job", "path", job.Path, "model", job.Model, "options", string(job.RawOptions), "audio_bytes", len(job.Audio))
		run(job, s)
	})
	w.Listener.Close()
	w.Listener = ln
	w.Start()
	log.Info("fakeworker listening", "addr", ln.Addr().String(), "script", *script)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	w.Close()
}
