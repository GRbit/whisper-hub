package workerclient

import "github.com/GRbit/whisper-hub/internal/core"

// Info is the body of GET /v1/info.
type Info struct {
	Engine       string            `json:"engine"`
	Version      string            `json:"version"`
	Device       string            `json:"device"`
	Models       []ModelInfo       `json:"models"`
	Languages    []string          `json:"languages"`
	Capabilities core.Capabilities `json:"capabilities"`
}

// ModelInfo is one entry of Info.Models.
type ModelInfo struct {
	ID      string `json:"id"`
	Default bool   `json:"default"`
	Loaded  bool   `json:"loaded"`
}

// DefaultModel returns the id of the model flagged as default, or the first
// model when the worker flagged none, or "" when it reports no models.
func (i *Info) DefaultModel() string {
	for _, m := range i.Models {
		if m.Default {
			return m.ID
		}
	}
	if len(i.Models) > 0 {
		return i.Models[0].ID
	}
	return ""
}

// Options is the JSON "options" part of POST /v1/transcribe. Pointer and
// empty-able fields are sent as null when unset, as the Worker API expects.
type Options struct {
	Model          string    `json:"model,omitempty"`
	Task           core.Task `json:"task"`
	Language       *string   `json:"language"`
	Prompt         *string   `json:"prompt"`
	Temperature    *float64  `json:"temperature"`
	VADFilter      bool      `json:"vad_filter"`
	WordTimestamps bool      `json:"word_timestamps"`
	Diarize        bool      `json:"diarize"`
	MinSpeakers    *int      `json:"min_speakers"`
	MaxSpeakers    *int      `json:"max_speakers"`
	LegacyOutput   *string   `json:"legacy_output"`
	// JobID names the job in POST /v1/pause; empty when it is never paused.
	JobID string `json:"job_id,omitempty"`
}

// OptionsFromRequest builds worker options for model from a core request.
// Empty strings become null, matching the Worker API defaults.
func OptionsFromRequest(model string, r *core.Request) Options {
	o := Options{
		Model:          model,
		Task:           r.Task,
		Temperature:    r.Temperature,
		VADFilter:      r.VADFilter,
		WordTimestamps: r.WordTimestamps,
		Diarize:        r.Diarize,
		MinSpeakers:    r.MinSpeakers,
		MaxSpeakers:    r.MaxSpeakers,
	}
	if o.Task == "" {
		o.Task = core.TaskTranscribe
	}
	if r.Language != "" {
		o.Language = &r.Language
	}
	if r.Prompt != "" {
		o.Prompt = &r.Prompt
	}
	if r.LegacyOutput != "" {
		o.LegacyOutput = &r.LegacyOutput
	}
	return o
}

// event is one NDJSON line of the /v1/transcribe response. Fields of all
// event types share one struct so a line is decoded once.
type event struct {
	Type string `json:"type"`
	// info
	Model               string  `json:"model"`
	Engine              string  `json:"engine"`
	Language            string  `json:"language"`
	LanguageProbability float64 `json:"language_probability"`
	Duration            float64 `json:"duration"`
	// segment
	Segment *core.Segment `json:"segment"`
	// done
	Text     string  `json:"text"`
	Rendered *string `json:"rendered"`
	// error
	Code    string `json:"code"`
	Message string `json:"message"`
}

type errorBody struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type detectBody struct {
	Language   string  `json:"language"`
	Confidence float64 `json:"confidence"`
	Model      string  `json:"model"`
	Engine     string  `json:"engine"`
}
