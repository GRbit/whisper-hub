package wyoming

// Event types this server reads or writes. Names and fields follow the
// wyoming Python package (asr.py, audio.py, info.py, error.py).
const (
	typeDescribe        = "describe"
	typeInfo            = "info"
	typeSelectProgram   = "select-program"
	typeTranscribe      = "transcribe"
	typeAudioStart      = "audio-start"
	typeAudioChunk      = "audio-chunk"
	typeAudioStop       = "audio-stop"
	typeTranscript      = "transcript"
	typeTranscriptStart = "transcript-start"
	typeTranscriptChunk = "transcript-chunk"
	typeTranscriptStop  = "transcript-stop"
	typeError           = "error"
	typePing            = "ping"
	typePong            = "pong"
)

type transcribeData struct {
	Name     *string `json:"name"`
	Language *string `json:"language"`
	// Accepted and ignored: Whisper has no endpointing to tune, and the
	// biasing lists would need a prompt policy we do not have yet.
	VADSensitivity  *string  `json:"vad_sensitivity"`
	TranscriptNames []string `json:"transcript_names"`
	TranscriptTerms []string `json:"transcript_terms"`
}

// audioFormat fields are pointers because clients disagree on which of them
// they send; missing ones fall back to the stream's or Wyoming's defaults.
type audioFormat struct {
	Rate     *int `json:"rate"`
	Width    *int `json:"width"`
	Channels *int `json:"channels"`
}

type selectProgramData struct {
	Name string `json:"name"`
}

type pingData struct {
	Text *string `json:"text"`
}

type transcriptData struct {
	Text     string `json:"text"`
	Language string `json:"language,omitempty"`
}

type transcriptStartData struct {
	Language string `json:"language,omitempty"`
}

type transcriptChunkData struct {
	Text string `json:"text"`
}

type errorData struct {
	Text string `json:"text"`
	Code string `json:"code,omitempty"`
}

type attribution struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

// asrModel and asrProgram serialize every field, nullable ones as null, the
// way the reference dataclasses' asdict does. Info.from_event in the Python
// package raises when a required field such as "installed" is missing.
type asrModel struct {
	Name        string      `json:"name"`
	Attribution attribution `json:"attribution"`
	Installed   bool        `json:"installed"`
	Description *string     `json:"description"`
	Version     *string     `json:"version"`
	Languages   []string    `json:"languages"`
}

type asrProgram struct {
	Name                         string      `json:"name"`
	Attribution                  attribution `json:"attribution"`
	Installed                    bool        `json:"installed"`
	Description                  *string     `json:"description"`
	Version                      *string     `json:"version"`
	Models                       []asrModel  `json:"models"`
	SupportsTranscriptStreaming  bool        `json:"supports_transcript_streaming"`
	RequiresExternalVAD          bool        `json:"requires_external_vad"`
	PrefersAutoGainEnabled       bool        `json:"prefers_auto_gain_enabled"`
	PrefersNoiseReductionEnabled bool        `json:"prefers_noise_reduction_enabled"`
}

// infoData lists every service array, empty ones included, as Info.event()
// in the reference package does.
type infoData struct {
	ASR    []asrProgram `json:"asr"`
	TTS    []struct{}   `json:"tts"`
	Handle []struct{}   `json:"handle"`
	Intent []struct{}   `json:"intent"`
	Wake   []struct{}   `json:"wake"`
	Mic    []struct{}   `json:"mic"`
	Snd    []struct{}   `json:"snd"`
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
