package legacy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/GRbit/whisper-hub/internal/core"
)

// The reference tests replay requests whose real responses were captured
// from the upstream service (testdata/reference/capture.sh runs
// TestCaptureReference against it) and compare status, headers and body
// with what this package answers in front of a fake core.Service.
//
// The fake service returns the reference's own transcript bytes as
// Result.Rendered (and its detected language for /detect-language), so a
// body match proves the frontend passes the worker's rendering through
// unchanged; the transcript itself is the worker's business.

const refBoundary = "zz"

// refHost is sent as the Host header on both sides: Starlette builds
// absolute redirect URLs from it.
const refHost = "asr.test:9000"

type refCase struct {
	name   string
	method string
	target string
	ctype  string
	body   func(t testing.TB) []byte
	// fail makes the fake service fail with this kind, standing in for the
	// engine exception the reference hit.
	fail *core.ErrorKind
}

func kind(k core.ErrorKind) *core.ErrorKind { return &k }

func readWAV(t testing.TB) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/audio/english-front-center.wav")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// wavData returns the samples of a RIFF/WAVE file, or ok=false.
func wavData(b []byte) ([]byte, bool) {
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, false
	}
	for p := 12; p+8 <= len(b); {
		id, n := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:p+8]))
		p += 8
		if id == "data" {
			return b[p:min(p+n, len(b))], true
		}
		p += n + n%2
	}
	return nil, false
}

func rawPCM(t testing.TB) []byte {
	d, _ := wavData(readWAV(t))
	return d
}

// headerOnlyWAV is a valid WAV with no samples: it decodes to 0 samples.
func headerOnlyWAV() []byte {
	b := make([]byte, 44)
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], 36)
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], 16000)
	binary.LittleEndian.PutUint32(b[28:], 32000)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	return b
}

type mpPart struct {
	headers string // raw header lines, each ending in \r\n
	data    []byte
}

func filePart(name, filename string, data []byte) mpPart {
	return mpPart{fmt.Sprintf("Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\nContent-Type: application/octet-stream\r\n", name, filename), data}
}

func fieldPart(name, value string) mpPart {
	return mpPart{fmt.Sprintf("Content-Disposition: form-data; name=\"%s\"\r\n", name), []byte(value)}
}

func multipartBody(parts ...mpPart) []byte {
	var b bytes.Buffer
	for _, p := range parts {
		b.WriteString("--" + refBoundary + "\r\n" + p.headers + "\r\n")
		b.Write(p.data)
		b.WriteString("\r\n")
	}
	b.WriteString("--" + refBoundary + "--\r\n")
	return b.Bytes()
}

const mpType = "multipart/form-data; boundary=" + refBoundary

func wavUpload(t testing.TB) []byte {
	return multipartBody(filePart("audio_file", "english-front-center.wav", readWAV(t)))
}

func asr(name, query string, fail ...core.ErrorKind) refCase {
	c := refCase{name: name, method: "POST", target: "/asr" + query, ctype: mpType, body: wavUpload}
	if len(fail) > 0 {
		c.fail = kind(fail[0])
	}
	return c
}

func raw(b []byte) func(testing.TB) []byte { return func(testing.TB) []byte { return b } }

func fileUpload(name string, data func(testing.TB) []byte) func(testing.TB) []byte {
	return func(t testing.TB) []byte { return multipartBody(filePart("audio_file", name, data(t))) }
}

var refCases = []refCase{
	asr("asr_default", ""),
	asr("asr_txt", "?output=txt"),
	asr("asr_vtt", "?output=vtt"),
	asr("asr_srt", "?output=srt"),
	asr("asr_tsv", "?output=tsv"),
	asr("asr_json", "?output=json"),
	asr("asr_json_words", "?output=json&word_timestamps=true&vad_filter=true"),
	asr("asr_language_en", "?language=en&task=transcribe"),
	asr("asr_language_empty", "?language=&initial_prompt="),
	asr("asr_language_bad", "?language=xx", core.KindBadRequest),
	asr("asr_task_bad", "?task=foo", core.KindBadRequest),
	asr("asr_output_unknown", "?output=foo"),
	asr("asr_output_empty", "?output="),
	asr("asr_output_latin1", "?output=%C3%BC"),
	asr("asr_output_not_latin1", "?output=%E2%82%AC"),
	asr("asr_dup_param", "?output=json&output=srt"),
	asr("asr_unknown_param", "?foo=bar&model=tiny"),
	asr("asr_bool_spellings", "?encode=YES&vad_filter=off&word_timestamps=F&diarize=n"),
	asr("asr_int_ok", "?min_speakers=+1_0&max_speakers=2.00"),
	asr("asr_bad_bool", "?encode=maybe"),
	asr("asr_bad_bool_empty", "?vad_filter=&word_timestamps=2&diarize=%20true"),
	asr("asr_bad_bool_unicode", "?encode=%C3%BC%0A%FF"),
	asr("asr_bad_int", "?min_speakers=x&max_speakers=1.5"),
	{name: "asr_many_errors", method: "POST", target: "/asr?encode=maybe&task=foo&min_speakers=x&output=json"},
	{name: "asr_no_body", method: "POST", target: "/asr"},
	{name: "asr_wrong_field", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte { return multipartBody(filePart("audio", "a.wav", readWAV(t))) }},
	{name: "asr_plain_field", method: "POST", target: "/asr", ctype: mpType,
		body: raw(multipartBody(fieldPart("audio_file", "hello \"x\"\n"), fieldPart("other", "y")))},
	{name: "asr_empty_field", method: "POST", target: "/asr", ctype: mpType,
		body: raw(multipartBody(fieldPart("audio_file", "")))},
	{name: "asr_urlencoded", method: "POST", target: "/asr", ctype: "application/x-www-form-urlencoded",
		body: raw([]byte("x=1&&audio_file=a+b%20c%FF"))},
	{name: "asr_json_body", method: "POST", target: "/asr", ctype: "application/json", body: raw([]byte(`{"audio_file":"x"}`))},
	{name: "asr_missing_boundary", method: "POST", target: "/asr", ctype: "multipart/form-data", body: raw([]byte("xx"))},
	{name: "asr_uppercase_ctype", method: "POST", target: "/asr", ctype: "Multipart/Form-Data; boundary=zz", body: wavUpload},
	{name: "asr_garbage_multipart", method: "POST", target: "/asr", ctype: mpType, body: raw([]byte("xx"))},
	{name: "asr_preamble", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte { return append([]byte("junk\r\n"), wavUpload(t)...) }},
	{name: "asr_leading_crlf", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte { return append([]byte("\r\n"), wavUpload(t)...) }},
	{name: "asr_no_name", method: "POST", target: "/asr", ctype: mpType,
		body: raw([]byte("--zz\r\nContent-Disposition: form-data; filename=\"a.wav\"\r\n\r\nabc\r\n--zz--\r\n"))},
	{name: "asr_bad_header", method: "POST", target: "/asr", ctype: mpType,
		body: raw([]byte("--zz\r\nContent-Disposition form-data\r\n\r\nabc\r\n--zz--\r\n"))},
	{name: "asr_truncated", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte {
			b := wavUpload(t)
			return b[:len(b)-200]
		}},
	{name: "asr_no_final_boundary", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte {
			b := wavUpload(t)
			return b[:len(b)-len("--\r\n")]
		}},
	{name: "asr_two_files", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte {
			return multipartBody(filePart("audio_file", "a.wav", readWAV(t)), filePart("audio_file", "b.bin", []byte("x13g2\n")))
		}},
	{name: "asr_file_then_field", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte {
			return multipartBody(filePart("audio_file", "a.wav", readWAV(t)), fieldPart("audio_file", "late"))
		}},
	{name: "asr_filename_special", method: "POST", target: "/asr", ctype: mpType,
		body: fileUpload(`my \"f\" `+"\u00fc\u20ac"+` ?#%&+.wav`, readWAV)},
	{name: "asr_filename_empty", method: "POST", target: "/asr", ctype: mpType, body: fileUpload("", readWAV)},
	{name: "asr_filename_ie_path", method: "POST", target: "/asr", ctype: mpType, body: fileUpload(`C:\dir\x.wav`, readWAV)},
	{name: "asr_filename_rel_path", method: "POST", target: "/asr", ctype: mpType, body: fileUpload("../a/b.wav", readWAV)},
	{name: "asr_filename_latin1", method: "POST", target: "/asr", ctype: mpType, body: fileUpload("\xfc.wav", readWAV)},
	{name: "asr_filename_unquoted", method: "POST", target: "/asr", ctype: mpType,
		body: func(t testing.TB) []byte {
			return multipartBody(mpPart{"Content-Disposition: attachment; name=audio_file; filename=a b.wav\r\n", readWAV(t)})
		}},
	{name: "asr_filename_rfc2231", method: "POST", target: "/asr", ctype: mpType,
		body: raw(multipartBody(mpPart{"Content-Disposition: form-data; name=\"audio_file\"; filename*=UTF-8''%C3%BC.wav\r\n", []byte("abc")}))},
	{name: "asr_empty_upload", method: "POST", target: "/asr", ctype: mpType, body: fileUpload("e.wav", raw(nil))},
	{name: "asr_garbage_upload", method: "POST", target: "/asr", ctype: mpType, body: fileUpload("g.wav", raw([]byte("this is not audio at all")))},
	{name: "asr_header_only_wav", method: "POST", target: "/asr", ctype: mpType, body: fileUpload("h.wav", raw(headerOnlyWAV()))},
	{name: "asr_encode_false", method: "POST", target: "/asr?encode=false&output=srt", ctype: mpType, body: fileUpload("a.pcm", rawPCM)},
	{name: "asr_encode_false_odd", method: "POST", target: "/asr?encode=false", ctype: mpType, body: fileUpload("a.pcm", raw([]byte{1, 2, 3}))},
	{name: "asr_encode_false_empty", method: "POST", target: "/asr?encode=0", ctype: mpType, body: fileUpload("a.pcm", raw(nil))},
	{name: "asr_get", method: "GET", target: "/asr"},
	{name: "asr_head", method: "HEAD", target: "/asr"},
	{name: "asr_options", method: "OPTIONS", target: "/asr"},
	{name: "asr_trailing_slash", method: "POST", target: "/asr/?output=json"},

	{name: "detect_ok", method: "POST", target: "/detect-language", ctype: mpType, body: wavUpload},
	{name: "detect_encode_false", method: "POST", target: "/detect-language?encode=false", ctype: mpType, body: fileUpload("a.pcm", rawPCM)},
	{name: "detect_missing", method: "POST", target: "/detect-language?encode=nope"},
	{name: "detect_garbage", method: "POST", target: "/detect-language", ctype: mpType, body: fileUpload("g.wav", raw([]byte("garbage")))},
	{name: "detect_get", method: "GET", target: "/detect-language"},

	{name: "root", method: "GET", target: "/?a=b"},
	{name: "root_head", method: "HEAD", target: "/"},
	{name: "root_post", method: "POST", target: "/"},
	{name: "docs", method: "GET", target: "/docs"},
	{name: "docs_head", method: "HEAD", target: "/docs"},
	{name: "docs_post", method: "POST", target: "/docs"},
	{name: "docs_slash", method: "GET", target: "/docs/"},
	{name: "docs_oauth2_redirect", method: "GET", target: "/docs/oauth2-redirect"},
	{name: "redoc", method: "GET", target: "/redoc"},
	{name: "openapi", method: "GET", target: "/openapi.json"},
	{name: "openapi_head", method: "HEAD", target: "/openapi.json"},
	{name: "openapi_delete", method: "DELETE", target: "/openapi.json"},
	{name: "assets_css", method: "GET", target: "/assets/swagger-ui.css"},
	{name: "assets_js", method: "GET", target: "/assets/swagger-ui-bundle.js"},
	{name: "assets_js_head", method: "HEAD", target: "/assets/swagger-ui-bundle.js"},
	{name: "assets_range", method: "GET", target: "/assets/swagger-ui-bundle.js", ctype: "range:bytes=0-10"},
	{name: "assets_not_modified", method: "GET", target: "/assets/swagger-ui.css", ctype: `inm:"1f75e3c45431eb214a976d3c64d2c497"`},
	{name: "assets_missing", method: "GET", target: "/assets/nope.js"},
	{name: "assets_dir", method: "GET", target: "/assets/"},
	{name: "assets_no_slash", method: "GET", target: "/assets"},
	{name: "assets_post", method: "POST", target: "/assets/swagger-ui.css"},
	{name: "unknown", method: "GET", target: "/nope"},
	{name: "unknown_post", method: "POST", target: "/v2/asr"},
	{name: "unknown_case", method: "POST", target: "/ASR"},
}

// capturedResponse is the on-disk form of one reference response.
type capturedResponse struct {
	Request struct {
		Method     string `json:"method"`
		Target     string `json:"target"`
		BodySHA256 string `json:"body_sha256,omitempty"`
	} `json:"request"`
	Status  int         `json:"status"`
	Headers [][2]string `json:"headers"`
	// Body is the response body when it is short UTF-8, else empty and
	// described by BodySHA256 and BodyLen.
	Body       string `json:"body"`
	BodySHA256 string `json:"body_sha256,omitempty"`
	BodyLen    int    `json:"body_len"`
}

func (c refCase) request(t testing.TB, base string) *http.Request {
	var body []byte
	if c.body != nil {
		body = c.body(t)
	}
	req, err := http.NewRequest(c.method, base+c.target, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Host = refHost
	switch {
	case strings.HasPrefix(c.ctype, "range:"):
		req.Header.Set("Range", strings.TrimPrefix(c.ctype, "range:"))
	case strings.HasPrefix(c.ctype, "inm:"):
		req.Header.Set("If-None-Match", strings.TrimPrefix(c.ctype, "inm:"))
	case c.ctype != "":
		req.Header.Set("Content-Type", c.ctype)
	}
	return req
}

func capture(t testing.TB, c refCase, base string) capturedResponse {
	client := &http.Client{
		Transport:     &http.Transport{DisableCompression: true},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	req := c.request(t, base)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s: %v", c.name, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("%s: %v", c.name, err)
	}
	var out capturedResponse
	out.Request.Method, out.Request.Target = c.method, c.target
	if c.body != nil {
		sum := sha256.Sum256(c.body(t))
		out.Request.BodySHA256 = hex.EncodeToString(sum[:])
	}
	out.Status = resp.StatusCode
	for k, vs := range resp.Header {
		for _, v := range vs {
			// Starlette sends latin-1 header bytes; keep them readable in JSON.
			if !utf8.ValidString(v) {
				v = "latin-1:" + latin1Decode([]byte(v))
			}
			out.Headers = append(out.Headers, [2]string{strings.ToLower(k), v})
		}
	}
	// Framing is not in resp.Header; record it so chunked vs length shows.
	if len(resp.TransferEncoding) > 0 {
		out.Headers = append(out.Headers, [2]string{"transfer-encoding", strings.Join(resp.TransferEncoding, ",")})
	}
	sort.Slice(out.Headers, func(i, j int) bool { return out.Headers[i][0] < out.Headers[j][0] })
	out.BodyLen = len(body)
	if len(body) <= 64<<10 && utf8.Valid(body) {
		out.Body = string(body)
	} else {
		sum := sha256.Sum256(body)
		out.BodySHA256 = hex.EncodeToString(sum[:])
	}
	return out
}

func casePath(name string) string {
	return filepath.Join("testdata", "reference", "cases", name+".json")
}

// TestCaptureReference records the reference responses. It only runs when
// LEGACY_REFERENCE_URL points at a running reference (see capture.sh).
func TestCaptureReference(t *testing.T) {
	base := os.Getenv("LEGACY_REFERENCE_URL")
	if base == "" {
		t.Skip("LEGACY_REFERENCE_URL not set")
	}
	if err := os.MkdirAll(filepath.Dir(casePath("x")), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, c := range refCases {
		got := capture(t, c, base)
		b, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(casePath(c.name), append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %d", c.name, got.Status)
	}
}

// ignoredHeaders differ by nature of the server, not by behavior.
var ignoredHeaders = map[string]string{
	"date":   "clock",
	"server": "uvicorn identifies itself; Go sends no Server header",
}

// unorderedHeaders hold a set whose order upstream does not fix: Starlette
// builds Allow from a Python set, so it varies per process. They are
// compared as sets.
var unorderedHeaders = map[string]bool{"allow": true}

// The recorded cases are public upstream's responses (capture.sh).
// whisper-hub deliberately answers some requests differently; each such
// difference is one entry below, and the expected response of a case is the
// recorded one with the applicable extensions applied. Nothing else may
// differ. README "Extensions to upstream" lists the same items.

// refAudioSeconds is the duration of the decoded test recording, as the
// X-Audio-Duration-Seconds header formats it.
func refAudioSeconds(t testing.TB) string {
	return fmt.Sprintf("%.3f", float64(len(rawPCM(t))/2)/core.SampleRate)
}

// Extension 3 and 4: undecodable or empty uploads get 400 with a redacted
// reason (size and leading bytes only) instead of upstream's 500 or, for
// audio without samples, its empty 200. The reasons follow from each
// case's upload.
var badAudioDetail = map[string]string{
	"asr_empty_upload":     "Failed to decode audio (0 bytes, magic=)",
	"asr_garbage_upload":   "Failed to decode audio (24 bytes, magic=74686973206973206e6f7420)",
	"asr_two_files":        "Failed to decode audio (6 bytes, magic=78313367320a)",
	"detect_garbage":       "Failed to decode audio (7 bytes, magic=67617262616765)",
	"asr_encode_false_odd": "Raw PCM payload has an odd length (3 bytes) (3 bytes, magic=010203)",
	// Extension 4: upstream answers these with an empty 200 transcript.
	"asr_header_only_wav":    "Audio decoded to 0 samples (44 bytes, magic=524946462400000057415645)",
	"asr_encode_false_empty": "Audio decoded to 0 samples (0 bytes, magic=)",
}

// withExtensions returns the response this package must give for a case
// whose recorded upstream response is want.
func withExtensions(t testing.TB, c refCase, want capturedResponse) capturedResponse {
	audioPath := c.method == "POST" && (strings.HasPrefix(c.target, "/asr?") || c.target == "/asr" ||
		strings.HasPrefix(c.target, "/detect-language"))
	detect := strings.HasPrefix(c.target, "/detect-language")
	if detail, ok := badAudioDetail[c.name]; ok {
		body := string(appendPyJSONString([]byte(`{"detail":`), detail)) + "}"
		return replaced(want, 400, "application/json", body, "")
	}
	if !audioPath || (want.Status != 200 && c.fail == nil) {
		return want
	}
	duration := refAudioSeconds(t)
	if c.fail != nil {
		// Extension 2: engine failures get a JSON detail, not Starlette's
		// plain-text 500, and still report the decoded duration.
		detail := "Transcription failed"
		if detect {
			detail = "Language detection failed"
		}
		return replaced(want, 500, "application/json", `{"detail":"`+detail+`"}`, duration)
	}
	// Extension 1: the decoded duration, as a header on every response that
	// decoded the upload, and in the /detect-language JSON body.
	want.Headers = append(want.Headers, [2]string{"x-audio-duration-seconds", duration})
	if detect {
		body := strings.TrimSuffix(want.Body, "}") + `,"audio_duration_seconds":` + duration + "}"
		return replaced(want, want.Status, "application/json", body, duration)
	}
	return want
}

// replaced is want with a new status and JSON or text body; framing headers
// follow the body. A non-empty duration adds the duration header.
func replaced(want capturedResponse, status int, ctype, body, duration string) capturedResponse {
	out := want
	out.Status, out.Body, out.BodyLen, out.BodySHA256 = status, body, len(body), ""
	out.Headers = nil
	for _, h := range want.Headers {
		switch h[0] {
		case "content-type", "content-length", "transfer-encoding", "content-disposition", "asr-engine", "x-audio-duration-seconds":
			continue
		}
		out.Headers = append(out.Headers, h)
	}
	out.Headers = append(out.Headers, [2]string{"content-type", ctype}, [2]string{"content-length", fmt.Sprint(len(body))})
	if duration != "" {
		out.Headers = append(out.Headers, [2]string{"x-audio-duration-seconds", duration})
	}
	return out
}

// Differences outside the recorded cases, by design or not reproduced:
//   - header names: uvicorn sends lowercase, Go canonical case; order differs
//   - Go ServeMux cleans paths ("//asr", "/assets/../x") with a 301; upstream 404s
//   - multipart charsets other than utf-8 and latin-1 are read as utf-8
//   - speaker counts beyond the int range saturate instead of staying big ints
//   - transcript bodies and detection values come from the worker, not checked here
//   - additive: the model query parameter (not in /openapi.json), 413 for
//     MaxUploadBytes, 401/403/404/503 for service kinds upstream lacks, and
//     Asr-Engine names the engine that ran rather than a configured one
//   - uvicorn/h11 transport behavior (limits, malformed HTTP, CR/LF in a
//     header value) is net/http's

func TestReference(t *testing.T) {
	for _, c := range refCases {
		t.Run(c.name, func(t *testing.T) {
			raw, err := os.ReadFile(casePath(c.name))
			if err != nil {
				t.Fatalf("no reference capture (run testdata/reference/capture.sh): %v", err)
			}
			var want capturedResponse
			if err := json.Unmarshal(raw, &want); err != nil {
				t.Fatal(err)
			}
			if c.body != nil {
				sum := sha256.Sum256(c.body(t))
				if hex.EncodeToString(sum[:]) != want.Request.BodySHA256 {
					t.Fatalf("request body changed since capture; re-run capture.sh")
				}
			}
			svc := newFakeService()
			svc.fail = c.fail
			svc.rendered = []byte(want.Body)
			if strings.HasPrefix(c.target, "/detect-language") && want.Status == 200 {
				var d struct {
					Code       string  `json:"language_code"`
					Confidence float64 `json:"confidence"`
				}
				if err := json.Unmarshal([]byte(want.Body), &d); err != nil {
					t.Fatal(err)
				}
				svc.detection = &core.Detection{Language: d.Code, Confidence: d.Confidence, Engine: "faster_whisper", Model: "tiny"}
			}
			mux := http.NewServeMux()
			New(Config{}, svc, fakeDecoder{}).Register(mux)
			srv := httptest.NewServer(mux)
			defer srv.Close()
			got := capture(t, c, srv.URL)
			compareCapture(t, c.name, withExtensions(t, c, want), got)
		})
	}
}

func compareCapture(t *testing.T, name string, want, got capturedResponse) {
	t.Helper()
	if got.Status != want.Status {
		t.Errorf("status = %d, want %d", got.Status, want.Status)
	}
	wh, gh := headerMap(want.Headers), headerMap(got.Headers)
	for k := range ignoredHeaders {
		delete(wh, k)
		delete(gh, k)
	}
	for k, v := range wh {
		if gv, ok := gh[k]; !ok {
			t.Errorf("header %s missing, want %q", k, v)
		} else if unorderedHeaders[k] && sameSet(gv, v) {
			continue
		} else if gv != v {
			t.Errorf("header %s = %q, want %q", k, gv, v)
		}
	}
	for k, v := range gh {
		if _, ok := wh[k]; !ok {
			t.Errorf("unexpected header %s: %q", k, v)
		}
	}
	if got.Body != want.Body || got.BodySHA256 != want.BodySHA256 || got.BodyLen != want.BodyLen {
		t.Errorf("body = %q (%d bytes, sha %s)\nwant %q (%d bytes, sha %s)",
			trunc(got.Body), got.BodyLen, got.BodySHA256, trunc(want.Body), want.BodyLen, want.BodySHA256)
	}
}

func sameSet(a, b string) bool {
	split := func(s string) []string {
		parts := strings.Split(s, ",")
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		sort.Strings(parts)
		return parts
	}
	return strings.Join(split(a), ",") == strings.Join(split(b), ",")
}

func trunc(s string) string {
	if len(s) > 400 {
		return s[:400] + "..."
	}
	return s
}

func headerMap(hs [][2]string) map[string]string {
	m := map[string]string{}
	for _, h := range hs {
		if old, ok := m[h[0]]; ok {
			m[h[0]] = old + "\n" + h[1]
		} else {
			m[h[0]] = h[1]
		}
	}
	return m
}

// fakeService stands in for the hub.
type fakeService struct {
	mu        sync.Mutex
	models    []core.Model
	fail      *core.ErrorKind
	err       error
	rendered  []byte
	detection *core.Detection
	requests  []*core.Request
	detects   []*core.DetectRequest
}

func newFakeService() *fakeService {
	return &fakeService{models: []core.Model{{ID: "tiny", Engine: "faster_whisper", Default: true, Aliases: []string{"whisper-1"}}}}
}

func (f *fakeService) failure() error {
	if f.err != nil {
		return f.err
	}
	if f.fail != nil {
		return core.WrapError(*f.fail, fmt.Errorf("engine exploded at /secret/path"), "engine failed")
	}
	return nil
}

func (f *fakeService) Transcribe(_ context.Context, req *core.Request, _ func(core.Segment) error) (*core.Result, error) {
	f.mu.Lock()
	f.requests = append(f.requests, req)
	f.mu.Unlock()
	if err := f.failure(); err != nil {
		return nil, err
	}
	return &core.Result{Model: "tiny", Engine: "faster_whisper", Duration: req.Audio.Seconds(), Rendered: f.rendered}, nil
}

func (f *fakeService) DetectLanguage(_ context.Context, req *core.DetectRequest) (*core.Detection, error) {
	f.mu.Lock()
	f.detects = append(f.detects, req)
	f.mu.Unlock()
	if err := f.failure(); err != nil {
		return nil, err
	}
	if f.detection == nil {
		return &core.Detection{Language: "en", Confidence: 0.5, Engine: "faster_whisper", Model: "tiny"}, nil
	}
	return f.detection, nil
}

func (f *fakeService) Models(context.Context) []core.Model { return f.models }

// fakeDecoder plays the runtime's ffmpeg decoder for WAV input. Its error
// text is the one the real decoder produces (upstream's detail).
type fakeDecoder struct{}

func (fakeDecoder) DecodeFile(_ context.Context, r io.Reader) (core.PCM, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, core.WrapError(core.KindBadAudio, err, "read failed")
	}
	if d, ok := wavData(b); ok {
		return core.PCM(d), nil
	}
	magic := b[:min(len(b), 12)]
	return nil, core.Errorf(core.KindBadAudio, "Failed to decode audio (%d bytes, magic=%s)", len(b), hex.EncodeToString(magic))
}

func (fakeDecoder) FromRaw(_ context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	if rate != core.SampleRate || width != 2 || channels != 1 {
		return nil, fmt.Errorf("fake decoder cannot convert %d/%d/%d", rate, width, channels)
	}
	return core.PCM(data), nil
}
