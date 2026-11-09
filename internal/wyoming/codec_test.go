package wyoming

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

var testLimits = limits{MaxHeaderBytes: 1024, MaxDataBytes: 256, MaxPayloadBytes: 64}

func TestCodecRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w := newFrameWriter(&buf)
	payload := []byte{1, 2, 3, 4}
	if err := w.WriteEvent(typeAudioChunk, map[string]any{"rate": 16000, "width": 2, "channels": 1, "timestamp": nil}, payload); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteEvent(typeTranscriptStop, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteEvent(typeTranscript, transcriptData{Text: "a < b & c"}, nil); err != nil {
		t.Fatal(err)
	}

	// Same layout as the Python writer: data in its own block, no inline data.
	first, _, _ := strings.Cut(buf.String(), "\n")
	var hdr map[string]any
	if err := json.Unmarshal([]byte(first), &hdr); err != nil {
		t.Fatal(err)
	}
	if _, ok := hdr["data"]; ok {
		t.Errorf("header carries inline data: %s", first)
	}
	if hdr["version"] != protocolVersion || hdr["payload_length"] != float64(4) {
		t.Errorf("unexpected header %s", first)
	}
	if !strings.Contains(buf.String(), `"a < b & c"`) {
		t.Errorf("HTML escaping on the wire: %q", buf.String())
	}

	r := newFrameReader(&buf, testLimits)
	ev, err := r.ReadEvent()
	if err != nil {
		t.Fatal(err)
	}
	var af audioFormat
	if err := ev.decodeData(&af); err != nil {
		t.Fatal(err)
	}
	if ev.Type != typeAudioChunk || *af.Rate != 16000 || *af.Width != 2 || *af.Channels != 1 || !bytes.Equal(ev.Payload, payload) {
		t.Errorf("got %+v %+v", ev, af)
	}
	ev, err = r.ReadEvent()
	if err != nil || ev.Type != typeTranscriptStop || ev.Data != nil || ev.Payload != nil {
		t.Errorf("stop: %+v %v", ev, err)
	}
	ev, err = r.ReadEvent()
	if err != nil {
		t.Fatal(err)
	}
	var td transcriptData
	if err := ev.decodeData(&td); err != nil || td.Text != "a < b & c" {
		t.Errorf("transcript: %+v %v", td, err)
	}
	if _, err := r.ReadEvent(); !errors.Is(err, io.EOF) {
		t.Errorf("want io.EOF at end, got %v", err)
	}
}

func TestReaderPythonFrameAndMerge(t *testing.T) {
	// Bytes as wyoming 1.11.0 async_write_event produces them, plus inline
	// header data that the data block must override (event.py merges with
	// dict.update).
	block := `{"name": "tiny", "language": "nl"}`
	frame := `{"type": "transcribe", "data": {"language": "en", "context": {"x": 1}}, "version": "1.11.0", "data_length": ` +
		itoa(len(block)) + "}\n" + block
	r := newFrameReader(strings.NewReader("\n\r\n"+frame), testLimits)
	ev, err := r.ReadEvent()
	if err != nil {
		t.Fatal(err)
	}
	var td transcribeData
	if err := ev.decodeData(&td); err != nil {
		t.Fatal(err)
	}
	if *td.Name != "tiny" || *td.Language != "nl" || ev.Data["context"] == nil {
		t.Errorf("merge wrong: %+v data=%v", td, ev.Data)
	}
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }

func TestReaderMalformed(t *testing.T) {
	long := strings.Repeat("x", 2000)
	cases := []struct {
		name     string
		in       string
		tooLarge bool
		want     error // nil means *frameError
	}{
		{name: "not json", in: "hello\n"},
		{name: "array header", in: "[1,2]\n"},
		{name: "missing type", in: `{"data":{}}` + "\n"},
		{name: "empty type", in: `{"type":""}` + "\n"},
		{name: "type not string", in: `{"type":5}` + "\n"},
		{name: "data not object", in: `{"type":"a","data":[1]}` + "\n"},
		{name: "trailing garbage", in: `{"type":"a"} x` + "\n"},
		{name: "negative data_length", in: `{"type":"a","data_length":-1}` + "\n"},
		{name: "float payload_length", in: `{"type":"a","payload_length":1.5}` + "\n"},
		{name: "exp payload_length", in: `{"type":"a","payload_length":1e1}` + "\n"},
		{name: "string payload_length", in: `{"type":"a","payload_length":"4"}` + "\n"},
		{name: "huge payload_length", in: `{"type":"a","payload_length":99999999999999999999999}` + "\n", tooLarge: true},
		{name: "data block not object", in: `{"type":"a","data_length":3}` + "\n" + `[1]`},
		{name: "header too long", in: `{"type":"` + long + `"}` + "\n", tooLarge: true},
		{name: "header too long without newline", in: long, tooLarge: true},
		{name: "data too long", in: `{"type":"a","data_length":257}` + "\n", tooLarge: true},
		{name: "payload too long", in: `{"type":"a","payload_length":65}` + "\n", tooLarge: true},
		{name: "truncated data", in: `{"type":"a","data_length":10}` + "\n{}", want: io.ErrUnexpectedEOF},
		{name: "truncated payload", in: `{"type":"a","payload_length":10}` + "\nabc", want: io.ErrUnexpectedEOF},
		{name: "truncated header", in: `{"type":"a"`, want: io.ErrUnexpectedEOF},
		{name: "empty stream", in: "", want: io.EOF},
		{name: "only blank lines", in: "\n\n  \n", want: io.EOF},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newFrameReader(strings.NewReader(tc.in), testLimits).ReadEvent()
			if tc.want != nil {
				if !errors.Is(err, tc.want) {
					t.Fatalf("want %v, got %v", tc.want, err)
				}
				return
			}
			var fe *frameError
			if !errors.As(err, &fe) {
				t.Fatalf("want *frameError, got %v", err)
			}
			if fe.TooLarge != tc.tooLarge {
				t.Fatalf("TooLarge = %v, want %v (%v)", fe.TooLarge, tc.tooLarge, fe)
			}
		})
	}
}

func TestReaderAcceptsNullAndZeroLengths(t *testing.T) {
	in := `{"type":"audio-stop","data":null,"data_length":0,"payload_length":0}` + "\n" +
		`{"type":"describe","data_length":null}` + "\n"
	r := newFrameReader(strings.NewReader(in), testLimits)
	for _, want := range []string{typeAudioStop, typeDescribe} {
		ev, err := r.ReadEvent()
		if err != nil || ev.Type != want {
			t.Fatalf("want %s, got %+v %v", want, ev, err)
		}
	}
}

func TestReaderLargeHeaderAcrossBuffer(t *testing.T) {
	// Longer than bufio's 64 KiB buffer, within the configured limit.
	text := strings.Repeat("a", 100<<10)
	in := `{"type":"transcript","data":{"text":"` + text + `"}}` + "\n"
	ev, err := newFrameReader(strings.NewReader(in), limits{MaxHeaderBytes: 200 << 10}).ReadEvent()
	if err != nil {
		t.Fatal(err)
	}
	var td transcriptData
	if err := ev.decodeData(&td); err != nil || len(td.Text) != len(text) {
		t.Fatalf("got %d chars, err %v", len(td.Text), err)
	}
}

// FuzzReader checks that arbitrary input never panics, that a successful
// read respects the limits, and that the reader terminates.
func FuzzReader(f *testing.F) {
	seeds := []string{
		`{"type":"describe"}` + "\n",
		`{"type":"audio-chunk","data":{"rate":16000,"width":2,"channels":1},"payload_length":4}` + "\nabcd",
		`{"type":"transcribe","data_length":17}` + "\n" + `{"language":"en"}`,
		`{"type":"a","data_length":-5,"payload_length":1e9}` + "\n",
		"\n\n{\n",
		`{"type":"a","data":{"x":` + strings.Repeat("[", 500) + "\n",
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		r := newFrameReader(bytes.NewReader(in), testLimits)
		for i := 0; i <= len(in); i++ {
			ev, err := r.ReadEvent()
			if err != nil {
				return
			}
			if ev.Type == "" {
				t.Fatal("event without type")
			}
			if len(ev.Payload) > testLimits.MaxPayloadBytes {
				t.Fatalf("payload %d over limit", len(ev.Payload))
			}
		}
		t.Fatal("reader produced more events than input bytes")
	})
}
