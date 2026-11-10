package workertest

import "github.com/GRbit/whisper-hub/internal/core"

// These types restate the Worker API wire format independently of the hub's
// client, so a drift on either side shows up as a test failure.

// Info is the body the fake serves on GET /v1/info.
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

// DefaultModel returns the id flagged default, else the first id, else "".
func (i Info) DefaultModel() string {
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

// Options is the decoded "options" part of a job. Unknown fields are
// rejected by the fake.
type Options struct {
	Model          string    `json:"model"`
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
	JobID          string    `json:"job_id"`
}

// PauseRequest is one POST /v1/pause the fake received.
type PauseRequest struct {
	JobID string `json:"job_id"`
	Model string `json:"model"`
}
