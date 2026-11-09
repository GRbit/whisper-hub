package openaiapi

import (
	"fmt"
	"math"
	"strings"

	"github.com/GRbit/whisper-hub/internal/core"
)

type usage struct {
	Type    string  `json:"type"`
	Seconds float64 `json:"seconds"`
}

// durationUsage reports whole seconds, rounded up, as whisper-1 bills.
func durationUsage(sec float64) *usage {
	return &usage{Type: "duration", Seconds: math.Ceil(sec)}
}

type jsonResponse struct {
	Text  string `json:"text"`
	Usage *usage `json:"usage,omitempty"`
}

type verboseSegment struct {
	ID               int     `json:"id"`
	Seek             int     `json:"seek"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int   `json:"tokens"`
	Temperature      float64 `json:"temperature"`
	AvgLogprob       float64 `json:"avg_logprob"`
	CompressionRatio float64 `json:"compression_ratio"`
	NoSpeechProb     float64 `json:"no_speech_prob"`
}

type verboseWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type verboseResponse struct {
	Task     string           `json:"task"`
	Language string           `json:"language"`
	Duration float64          `json:"duration"`
	Text     string           `json:"text"`
	Segments []verboseSegment `json:"segments,omitempty"`
	Words    []verboseWord    `json:"words,omitempty"`
	Usage    *usage           `json:"usage,omitempty"`
}

// fullText is the transcript as OpenAI returns it: no leading space from
// the first Whisper token.
func fullText(res *core.Result) string {
	if res.Text != "" {
		return strings.TrimSpace(res.Text)
	}
	var b strings.Builder
	for _, s := range res.Segments {
		b.WriteString(s.Text)
	}
	return strings.TrimSpace(b.String())
}

func buildVerbose(res *core.Result, p *params, task core.Task, audioSec float64) *verboseResponse {
	v := &verboseResponse{
		Task:     string(task),
		Language: core.LanguageName(res.Language),
		Duration: res.Duration,
		Text:     fullText(res),
	}
	if v.Duration == 0 {
		v.Duration = audioSec
	}
	if task == core.TaskTranslate {
		// The translation output language, documented as always english.
		v.Language = "english"
	} else {
		v.Usage = durationUsage(v.Duration)
	}
	if v.Language == "" {
		v.Language = res.Language
	}
	if p.granSegment || task == core.TaskTranslate {
		v.Segments = make([]verboseSegment, 0, len(res.Segments))
		for _, s := range res.Segments {
			tokens := s.Tokens
			if tokens == nil {
				tokens = []int{}
			}
			v.Segments = append(v.Segments, verboseSegment{
				ID: s.ID, Seek: s.Seek, Start: s.Start, End: s.End, Text: s.Text,
				Tokens: tokens, Temperature: s.Temperature, AvgLogprob: s.AvgLogprob,
				CompressionRatio: s.CompressionRatio, NoSpeechProb: s.NoSpeechProb,
			})
		}
	}
	if p.granWord && task == core.TaskTranscribe {
		v.Words = []verboseWord{}
		for _, s := range res.Segments {
			for _, w := range s.Words {
				v.Words = append(v.Words, verboseWord{Word: strings.TrimSpace(w.Word), Start: w.Start, End: w.End})
			}
		}
	}
	return v
}

// cueTime formats seconds as HH:MM:SS<sep>mmm.
func cueTime(sec float64, sep string) string {
	ms := int64(math.Round(sec * 1000))
	if ms < 0 {
		ms = 0
	}
	h := ms / 3600000
	m := ms / 60000 % 60
	s := ms / 1000 % 60
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", h, m, s, sep, ms%1000)
}

func renderSRT(segs []core.Segment) string {
	var b strings.Builder
	for i, s := range segs {
		fmt.Fprintf(&b, "%d\n%s --> %s\n%s\n\n", i+1, cueTime(s.Start, ","), cueTime(s.End, ","), strings.TrimSpace(s.Text))
	}
	return b.String()
}

func renderVTT(segs []core.Segment) string {
	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for _, s := range segs {
		fmt.Fprintf(&b, "%s --> %s\n%s\n\n", cueTime(s.Start, "."), cueTime(s.End, "."), strings.TrimSpace(s.Text))
	}
	return b.String()
}
