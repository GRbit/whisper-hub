package wyoming

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
)

// protocolVersion is the Wyoming version this package implements and sends
// in every event header.
//
// 1.11.0 matters: from that version on, transcript-chunk carries the whole
// transcript so far instead of a fragment to append.
const protocolVersion = "1.11.0"

// event is one decoded Wyoming event.
type event struct {
	Type string
	// Data is the header data with the additional data block merged on top,
	// as the reference implementation does.
	Data    map[string]json.RawMessage
	Payload []byte
}

// decodeData unmarshals the event data into v.
func (e *event) decodeData(v any) error {
	if len(e.Data) == 0 {
		return nil
	}
	raw, err := json.Marshal(e.Data)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("decode %s data: %w", e.Type, err)
	}
	return nil
}

// limits bounds what a frameReader accepts from a peer.
type limits struct {
	MaxHeaderBytes  int // JSON header line, newline included
	MaxDataBytes    int // additional data block
	MaxPayloadBytes int // binary payload
}

// frameError reports a frame that violates the wire format or the limits.
// The stream cannot be resynchronized after one.
type frameError struct {
	Msg string
	// TooLarge is set when a limit was exceeded rather than the syntax broken.
	TooLarge bool
}

// Error implements the error interface.
func (e *frameError) Error() string { return "wyoming: " + e.Msg }

func frameErrorf(format string, args ...any) *frameError {
	return &frameError{Msg: fmt.Sprintf(format, args...)}
}

func tooLargef(format string, args ...any) *frameError {
	return &frameError{Msg: fmt.Sprintf(format, args...), TooLarge: true}
}

// frameReader decodes Wyoming events from a byte stream.
type frameReader struct {
	br  *bufio.Reader
	lim limits
}

// newFrameReader returns a Reader enforcing lim. Non-positive limits mean no limit,
// which is only sensible for trusted peers.
func newFrameReader(r io.Reader, lim limits) *frameReader {
	return &frameReader{br: bufio.NewReaderSize(r, 64<<10), lim: lim}
}

type wireHeader struct {
	Type          *string         `json:"type"`
	Data          json.RawMessage `json:"data"`
	DataLength    json.RawMessage `json:"data_length"`
	PayloadLength json.RawMessage `json:"payload_length"`
}

// ReadEvent reads the next event. It returns io.EOF when the stream ends
// cleanly between events, io.ErrUnexpectedEOF when it ends inside one, and a
// *frameError for malformed or oversized frames.
func (r *frameReader) ReadEvent() (*event, error) {
	line, err := r.readHeaderLine()
	if err != nil {
		return nil, err
	}

	var h wireHeader
	if err := json.Unmarshal(line, &h); err != nil {
		return nil, frameErrorf("invalid header JSON: %v", err)
	}
	if h.Type == nil || *h.Type == "" {
		return nil, frameErrorf("header has no event type")
	}
	ev := &event{Type: *h.Type}

	if len(h.Data) > 0 && !bytes.Equal(h.Data, []byte("null")) {
		if err := json.Unmarshal(h.Data, &ev.Data); err != nil {
			return nil, frameErrorf("header data of %q is not a JSON object", ev.Type)
		}
	}

	dataLen, err := parseLength(h.DataLength, "data_length", r.lim.MaxDataBytes)
	if err != nil {
		return nil, err
	}
	payloadLen, err := parseLength(h.PayloadLength, "payload_length", r.lim.MaxPayloadBytes)
	if err != nil {
		return nil, err
	}

	if dataLen > 0 {
		block := make([]byte, dataLen)
		if _, err := io.ReadFull(r.br, block); err != nil {
			return nil, unexpected(err)
		}
		var extra map[string]json.RawMessage
		if err := json.Unmarshal(block, &extra); err != nil {
			return nil, frameErrorf("data block of %q is not a JSON object", ev.Type)
		}
		if ev.Data == nil {
			ev.Data = extra
		} else {
			for k, v := range extra {
				ev.Data[k] = v
			}
		}
	}

	if payloadLen > 0 {
		ev.Payload = make([]byte, payloadLen)
		if _, err := io.ReadFull(r.br, ev.Payload); err != nil {
			return nil, unexpected(err)
		}
	}
	return ev, nil
}

// readHeaderLine returns the next non-blank line without its terminator.
// Blank lines are skipped: they carry nothing and some hand-written clients
// emit them.
func (r *frameReader) readHeaderLine() ([]byte, error) {
	for {
		var line []byte
		for {
			chunk, err := r.br.ReadSlice('\n')
			line = append(line, chunk...)
			if r.lim.MaxHeaderBytes > 0 && len(line) > r.lim.MaxHeaderBytes {
				return nil, tooLargef("header line exceeds %d bytes", r.lim.MaxHeaderBytes)
			}
			if err == nil {
				break
			}
			if errors.Is(err, bufio.ErrBufferFull) {
				continue
			}
			if errors.Is(err, io.EOF) {
				if len(bytes.TrimSpace(line)) == 0 {
					return nil, io.EOF
				}
				return nil, io.ErrUnexpectedEOF
			}
			return nil, err
		}
		if trimmed := bytes.TrimSpace(line); len(trimmed) > 0 {
			return trimmed, nil
		}
	}
}

// parseLength takes the raw JSON so that a quoted "10", which json.Number
// would accept, is rejected like the reference reader rejects it.
func parseLength(raw json.RawMessage, name string, limit int) (int, error) {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return 0, nil
	}
	v, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil {
		var ne *strconv.NumError
		if errors.As(err, &ne) && errors.Is(ne.Err, strconv.ErrRange) {
			return 0, tooLargef("%s does not fit in 64 bits", name)
		}
		return 0, frameErrorf("%s is not an integer: %.32s", name, raw)
	}
	if v < 0 {
		return 0, frameErrorf("%s is negative: %d", name, v)
	}
	if limit > 0 && v > int64(limit) {
		return 0, tooLargef("%s %d exceeds limit %d", name, v, limit)
	}
	if v > int64(maxInt) {
		return 0, tooLargef("%s %d does not fit in memory", name, v)
	}
	return int(v), nil
}

const maxInt = int(^uint(0) >> 1)

func unexpected(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// frameWriter encodes Wyoming events. It is not safe for concurrent use.
type frameWriter struct {
	w io.Writer
}

// newFrameWriter returns a Writer on w.
func newFrameWriter(w io.Writer) *frameWriter { return &frameWriter{w: w} }

type outHeader struct {
	Type          string `json:"type"`
	Version       string `json:"version"`
	DataLength    int    `json:"data_length,omitempty"`
	PayloadLength int    `json:"payload_length,omitempty"`
}

// WriteEvent writes one event. data must marshal to a JSON object or null;
// like the reference writer it goes into a separate data block, and an empty
// object is omitted.
func (w *frameWriter) WriteEvent(eventType string, data any, payload []byte) error {
	var dataBytes []byte
	if data != nil {
		b, err := marshalNoEscape(data)
		if err != nil {
			return fmt.Errorf("encode %s data: %w", eventType, err)
		}
		if !bytes.Equal(b, []byte("null")) && !bytes.Equal(b, []byte("{}")) {
			dataBytes = b
		}
	}
	hdr, err := marshalNoEscape(outHeader{
		Type:          eventType,
		Version:       protocolVersion,
		DataLength:    len(dataBytes),
		PayloadLength: len(payload),
	})
	if err != nil {
		return err
	}
	// One Write per event keeps frames whole on the wire and halves syscalls.
	buf := make([]byte, 0, len(hdr)+1+len(dataBytes)+len(payload))
	buf = append(buf, hdr...)
	buf = append(buf, '\n')
	buf = append(buf, dataBytes...)
	buf = append(buf, payload...)
	_, err = w.w.Write(buf)
	return err
}

// marshalNoEscape mirrors Python's json.dumps(ensure_ascii=False): no HTML
// escaping, so transcripts with "<" or "&" stay readable on the wire.
func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
