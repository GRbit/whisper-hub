package wyoming

import (
	"slices"
	"strings"

	"github.com/GRbit/whisper-hub/internal/core"
)

// cantoneseTag is advertised next to Whisper's "yue": Home Assistant's
// language matcher treats yue and zh as unrelated, and its intents ship
// zh-HK but no bare yue, so a Cantonese pipeline could never select "yue".
// wyoming-faster-whisper 3.9.0 made the same change.
const cantoneseTag = "zh-HK"

// advertisedLanguages is the language list shown in info for one model.
func advertisedLanguages(m core.Model) []string {
	langs := m.Languages
	if len(langs) == 0 {
		// An empty list would leave Home Assistant with nothing to pick, and
		// every Whisper model is multilingual unless the worker says otherwise.
		langs = core.Languages()
	}
	out := make([]string, 0, len(langs)+1)
	hasYue, hasTag := false, false
	for _, l := range langs {
		out = append(out, l)
		switch strings.ToLower(l) {
		case "yue":
			hasYue = true
		case strings.ToLower(cantoneseTag):
			hasTag = true
		}
	}
	if hasYue && !hasTag {
		out = append(out, cantoneseTag)
	}
	return out
}

// resolveLanguage maps the language a client asked for onto a Whisper code
// the model supports, or "" for auto-detection.
//
// Home Assistant normally echoes one of the advertised codes, but falls back
// to the raw pipeline language (e.g. "pt-BR") when its matcher found nothing
// or the pipeline has no stt_language; anything unusable means auto-detect
// rather than a failed request.
func resolveLanguage(requested string, supported []string) string {
	tag := strings.ToLower(strings.TrimSpace(requested))
	tag = strings.ReplaceAll(tag, "_", "-")
	if tag == "" || tag == "auto" {
		return ""
	}
	if len(supported) == 0 {
		supported = core.Languages()
	}
	has := func(code string) string {
		for _, s := range supported {
			if strings.EqualFold(s, code) {
				return s
			}
		}
		return ""
	}
	if tag == strings.ToLower(cantoneseTag) && has("yue") != "" {
		return has("yue")
	}
	if s := has(tag); s != "" {
		return s
	}
	if base, _, ok := strings.Cut(tag, "-"); ok {
		if s := has(base); s != "" {
			return s
		}
	}
	return ""
}

// modelFor returns the model a request for name would most likely run: the
// listed model with that id or alias, else the default. The Service does the
// real resolution; this only picks the language list to match against.
func modelFor(models []core.Model, name string) (core.Model, bool) {
	for _, m := range models {
		if m.ID == name || slices.Contains(m.Aliases, name) {
			return m, true
		}
	}
	for _, m := range models {
		if m.Default {
			return m, true
		}
	}
	if len(models) > 0 {
		return models[0], true
	}
	return core.Model{}, false
}
