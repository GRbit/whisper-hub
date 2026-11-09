package openaiapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Response formats accepted by whisper-1.
const (
	formatJSON        = "json"
	formatText        = "text"
	formatSRT         = "srt"
	formatVTT         = "vtt"
	formatVerboseJSON = "verbose_json"
	formatDiarized    = "diarized_json"
)

const supportedFormats = "'json', 'text', 'srt', 'verbose_json', 'vtt'"

// maxFieldBytes caps every non-file form field; a prompt is at most a few KB.
const maxFieldBytes = 1 << 20

// params is a parsed and validated transcription or translation request.
type params struct {
	audio       core.PCM
	hasFile     bool
	model       string
	language    string
	languages   []string
	prompt      string
	temperature *float64
	format      string
	granSegment bool
	granWord    bool
	granGiven   bool
	stream      bool
}

// sizeTracker remembers whether the body limit was hit, because the decoder
// may wrap or replace the *http.MaxBytesError it sees.
type sizeTracker struct {
	r       io.Reader
	tripped bool
}

func (t *sizeTracker) Read(p []byte) (int, error) {
	n, err := t.r.Read(p)
	var mbe *http.MaxBytesError
	if err != nil && errors.As(err, &mbe) {
		t.tripped = true
	}
	return n, err
}

func (s *Server) tooLarge(read int64) *apiError {
	msg := fmt.Sprintf("Maximum content size limit (%d) exceeded.", s.cfg.MaxUploadBytes)
	if read > 0 {
		msg = fmt.Sprintf("Maximum content size limit (%d) exceeded (%d bytes read).", s.cfg.MaxUploadBytes, read)
	}
	return &apiError{Status: http.StatusRequestEntityTooLarge, Message: msg, Type: typeServer}
}

// parseForm reads the multipart body field by field and decodes the file
// part as it streams in, so a long upload is never buffered twice. Fields
// are validated on arrival so a bad parameter sent before the file costs no
// decoding.
func (s *Server) parseForm(ctx context.Context, w http.ResponseWriter, r *http.Request, task core.Task, log *slog.Logger) (*params, error) {
	if s.cfg.MaxUploadBytes > 0 && r.ContentLength > s.cfg.MaxUploadBytes {
		return nil, s.tooLarge(r.ContentLength)
	}
	mt, mparams, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mt != "multipart/form-data" || mparams["boundary"] == "" {
		return nil, badRequest("", "", "Invalid Content-Type header (%s), expected multipart/form-data.", r.Header.Get("Content-Type"))
	}
	body := io.Reader(r.Body)
	tracker := &sizeTracker{r: body}
	if s.cfg.MaxUploadBytes > 0 {
		tracker.r = http.MaxBytesReader(w, r.Body, s.cfg.MaxUploadBytes)
	}
	mr := multipart.NewReader(tracker, mparams["boundary"])

	p := &params{format: formatJSON, granSegment: true}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			if tracker.tripped {
				return nil, s.tooLarge(0)
			}
			return nil, badRequest("", "", "Could not parse multipart form: %v", err)
		}
		name := part.FormName()
		if name == "file" {
			if p.hasFile {
				log.Debug("openai: ignoring extra file part")
				_, _ = io.Copy(io.Discard, part)
				continue
			}
			p.hasFile = true
			log.Debug("openai: decoding upload", "filename", part.FileName(), "content_type", part.Header.Get("Content-Type"))
			pcm, err := s.dec.DecodeFile(ctx, part)
			if tracker.tripped {
				return nil, s.tooLarge(0)
			}
			if err != nil {
				return nil, err
			}
			p.audio = pcm
			log.Debug("openai: upload decoded", "audio_seconds", pcm.Seconds())
			continue
		}
		val, err := io.ReadAll(io.LimitReader(part, maxFieldBytes+1))
		if err != nil {
			if tracker.tripped {
				return nil, s.tooLarge(0)
			}
			return nil, badRequest("", "", "Could not parse multipart form: %v", err)
		}
		if len(val) > maxFieldBytes {
			return nil, badRequest(name, "", "Form field '%s' is too large.", name)
		}
		if err := p.setField(name, string(val), task, log); err != nil {
			return nil, err
		}
	}
	if !p.hasFile {
		return nil, badRequest("file", "missing_required_parameter", "Missing required parameter: 'file'.")
	}
	if p.language == "" && len(p.languages) == 1 {
		p.language = p.languages[0]
	}
	return p, nil
}

// setField applies one form field. Unknown fields are ignored, as the real
// API does for most parameters it does not use.
func (p *params) setField(name, val string, task core.Task, log *slog.Logger) error {
	translate := task == core.TaskTranslate
	switch {
	case name == "model":
		p.model = val
	case name == "prompt":
		p.prompt = val
	case name == "response_format":
		switch val {
		case formatJSON, formatText, formatSRT, formatVTT, formatVerboseJSON:
			p.format = val
		case formatDiarized:
			return badRequest("response_format", "unsupported_value",
				"response_format 'diarized_json' is not supported: speaker diarization is not available for whisper models.")
		case "":
		default:
			return invalidValue("response_format", val, supportedFormats)
		}
	case name == "temperature":
		if val == "" {
			return nil
		}
		t, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return badRequest("temperature", "invalid_type", "Invalid type for 'temperature': expected a decimal, but got a string instead.")
		}
		if t < 0 || t > 1 {
			return badRequest("temperature", "decimal_out_of_range",
				"Invalid 'temperature': decimal out of range. Expected a value between 0 and 1, but got %s instead.", val)
		}
		// OpenAI documents 0 as "raise the temperature automatically on
		// failure", which is the engines' default fallback schedule.
		if t != 0 {
			p.temperature = &t
		}
	case translate && (name == "language" || name == "stream" ||
		name == "timestamp_granularities[]" || name == "timestamp_granularities"):
		log.Debug("openai: ignoring field not supported by translations", "field", name)
	case name == "language":
		if val == "" {
			return nil
		}
		code, err := checkLanguage("language", val)
		if err != nil {
			return err
		}
		p.language = code
	case name == "languages[]" || name == "languages":
		for _, v := range splitList(val) {
			code, err := checkLanguage("languages", v)
			if err != nil {
				return err
			}
			p.languages = append(p.languages, code)
		}
	case name == "timestamp_granularities[]" || name == "timestamp_granularities":
		if !p.granGiven {
			p.granGiven = true
			p.granSegment = false
		}
		for _, v := range splitList(val) {
			switch v {
			case "segment":
				p.granSegment = true
			case "word":
				p.granWord = true
			default:
				return invalidValue("timestamp_granularities", v, "'word', 'segment'")
			}
		}
	case name == "stream":
		if val == "" {
			return nil
		}
		b, err := strconv.ParseBool(val)
		if err != nil {
			return badRequest("stream", "invalid_type", "Invalid type for 'stream': expected a boolean, but got a string instead.")
		}
		p.stream = b
	case name == "include[]" || name == "include" || strings.HasPrefix(name, "chunking_strategy") ||
		name == "known_speaker_names[]" || name == "known_speaker_references[]" ||
		name == "keywords[]" || name == "keywords":
		// gpt-* only knobs that change no contract we can keep: no logprobs,
		// no server-side chunking, no keyword boosting.
		log.Debug("openai: ignoring gpt-only field", "field", name)
	default:
		log.Debug("openai: ignoring unknown field", "field", name)
	}
	return nil
}

// splitList accepts both one value per field and a comma separated list,
// since clients that do not use the [] spelling sometimes join values.
func splitList(val string) []string {
	var out []string
	for _, v := range strings.Split(val, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func checkLanguage(param, val string) (string, error) {
	code := strings.ToLower(strings.TrimSpace(val))
	if core.LanguageName(code) == "" {
		return "", badRequest(param, "invalid_language_format",
			"Invalid language '%s'. Language parameter must be specified in ISO-639-1 format.", val)
	}
	return code, nil
}
