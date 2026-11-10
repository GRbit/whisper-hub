// Package core holds the contracts shared by every part of whisper-hub.
//
// Protocol frontends (legacy /asr, OpenAI, Wyoming) depend only on this
// package, never on each other or on the runtime, so each can be built and
// tested against a fake Service.
package core

import (
	"context"
	"io"
	"sort"
	"time"
)

// SampleRate is the only sample rate the engines accept.
const SampleRate = 16000

// PCM is mono signed 16-bit little-endian audio at SampleRate.
type PCM []byte

// Seconds returns the audio length in seconds.
func (p PCM) Seconds() float64 { return float64(len(p)/2) / SampleRate }

// Duration returns the audio length.
func (p PCM) Duration() time.Duration {
	return time.Duration(len(p)/2) * time.Second / SampleRate
}

// Task selects between same-language transcription and translation to English.
type Task string

// Supported tasks.
const (
	TaskTranscribe Task = "transcribe"
	TaskTranslate  Task = "translate"
)

// Protocol names used in Request.Protocol, logs and usage records.
const (
	ProtocolLegacy  = "legacy"
	ProtocolOpenAI  = "openai"
	ProtocolWyoming = "wyoming"
)

// Request is one transcription job.
type Request struct {
	Audio PCM
	// Model is the public model id or alias as the client sent it; empty
	// means the default model. The Service resolves it and enforces access.
	Model       string
	Task        Task
	Language    string // Whisper language code; empty = auto-detect
	Prompt      string
	Temperature *float64 // nil = engine default
	VADFilter   bool
	// WordTimestamps asks for Segment.Words to be filled.
	WordTimestamps bool
	Diarize        bool
	MinSpeakers    *int
	MaxSpeakers    *int
	// LegacyOutput, when set to txt, vtt, srt, tsv or json, makes the engine
	// worker also render the transcript exactly as the reference
	// whisper-asr-webservice would for its engine; see Result.Rendered.
	LegacyOutput string
	// Protocol is one of the Protocol constants. With the caller's identity
	// it decides the job's priority; clients cannot choose one.
	Protocol string
	// Endpoint is the client-facing operation, e.g. "/asr" or
	// "/v1/audio/transcriptions"; it only feeds logs and usage records.
	Endpoint string
}

// Word is one word with timing, present when word timestamps were requested.
type Word struct {
	Word        string  `json:"word"`
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Probability float64 `json:"probability"`
	Speaker     string  `json:"speaker,omitempty"`
}

// Segment is one transcript segment in engine-neutral form.
type Segment struct {
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
	Words            []Word  `json:"words,omitempty"`
	Speaker          string  `json:"speaker,omitempty"`
}

// Result is the outcome of a finished transcription.
type Result struct {
	Model               string // model id that actually ran
	Engine              string // faster_whisper, whisperx, openai_whisper, ...
	Language            string // detected or requested language code
	LanguageProbability float64
	Duration            float64 // audio seconds
	Text                string
	Segments            []Segment
	// Rendered is the byte-exact legacy body, set only when
	// Request.LegacyOutput was given.
	Rendered []byte
}

// DetectRequest asks for the spoken language of a clip.
type DetectRequest struct {
	Audio    PCM
	Model    string
	Protocol string
	Endpoint string
}

// Detection is the result of language detection.
type Detection struct {
	Language   string // Whisper language code
	Confidence float64
	Engine     string
	Model      string
}

// Capabilities lists the optional features a model's engine supports.
type Capabilities struct {
	WordTimestamps bool `json:"word_timestamps"`
	VADFilter      bool `json:"vad_filter"`
	Diarize        bool `json:"diarize"`
	Translate      bool `json:"translate"`
}

// Model describes one model reachable through the hub.
type Model struct {
	ID string
	// Aliases are the extra names this model answers to, e.g. "whisper-1".
	// OpenAI clients look models up by these names, so they must be listed.
	Aliases      []string
	Engine       string
	Languages    []string // Whisper language codes
	Default      bool
	Capabilities Capabilities
}

// Service is the single entry point protocol frontends use to run speech
// recognition. Implementations resolve aliases, enforce per-user model
// access using the Identity in ctx, schedule by the priority that identity
// and Request.Protocol resolve to, and record usage, so frontends must not
// repeat any of that.
type Service interface {
	// Transcribe runs req and returns the complete result. When onSegment
	// is not nil it is called for each segment as soon as the engine
	// produces it; an error from it cancels the job. Errors are *Error.
	Transcribe(ctx context.Context, req *Request, onSegment func(Segment) error) (*Result, error)
	// DetectLanguage identifies the spoken language of the audio.
	DetectLanguage(ctx context.Context, req *DetectRequest) (*Detection, error)
	// Models lists the models the Identity in ctx may use, default first.
	Models(ctx context.Context) []Model
}

// AudioDecoder turns client audio into PCM.
type AudioDecoder interface {
	// DecodeFile decodes any container or codec ffmpeg understands.
	DecodeFile(ctx context.Context, r io.Reader) (PCM, error)
	// FromRaw converts headerless signed little-endian PCM (width is bytes
	// per sample) to mono SampleRate, resampling and downmixing as needed.
	FromRaw(ctx context.Context, data []byte, rate, width, channels int) (PCM, error)
}

// Languages returns all Whisper language codes, sorted.
func Languages() []string {
	codes := make([]string, 0, len(whisperLanguages))
	for _, l := range whisperLanguages {
		codes = append(codes, l.Code)
	}
	sort.Strings(codes)
	return codes
}

// LanguageName returns the lowercase English name of a Whisper language
// code ("en" -> "english"), or "" when the code is unknown.
func LanguageName(code string) string {
	for _, l := range whisperLanguages {
		if l.Code == code {
			return l.Name
		}
	}
	return ""
}
