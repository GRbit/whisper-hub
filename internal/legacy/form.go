package legacy

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"strings"
	"unicode/utf8"
)

// This file is a streaming port of the form parsing the reference service
// gets from Starlette and python-multipart. mime/multipart is not used
// because it differs in observable ways: it accepts a preamble, decodes
// RFC 2231 filename*, applies filepath.Base to filenames and rejects
// unquoted parameters that python-multipart accepts.

const (
	maxPartSize     = 1024 * 1024 // Starlette max_part_size for non-file parts
	maxFormFiles    = 1000
	maxFormFields   = 1000
	maxHeaderCount  = 8
	maxHeaderSize   = 4096 + 128
	maxBoundaryLen  = 256
	formBufferBytes = 64 << 10
)

// errBodyParse is any python-multipart parse error; FastAPI turns it into
// 400 "There was an error parsing the body".
var errBodyParse = errors.New("There was an error parsing the body")

// errTruncated means the body ended inside a part. python-multipart's
// finalize() accepts that silently: the unfinished part is simply dropped.
var errTruncated = errors.New("form body truncated")

// formError is Starlette's MultiPartException: 400 with its message as
// the detail.
type formError struct{ msg string }

func (e *formError) Error() string { return e.msg }

// bodyError wraps a failure to read the request body itself (client gone,
// size limit), as opposed to a malformed body.
type bodyError struct{ err error }

func (e *bodyError) Error() string { return e.err.Error() }
func (e *bodyError) Unwrap() error { return e.err }

// parseOptionsHeader is python-multipart's parse_options_header.
func parseOptionsHeader(value string) (string, map[string]string) {
	if value == "" {
		return "", map[string]string{}
	}
	if !strings.Contains(value, ";") {
		return strings.ToLower(strings.TrimSpace(value)), map[string]string{}
	}
	segs := parseParam(value)
	opts := map[string]string{}
	for _, seg := range segs[1:] {
		key, val, _ := strings.Cut(seg, "=")
		// RFC 7578 forbids key*= in form-data; python-multipart skips it.
		if strings.Contains(key, "*") {
			continue
		}
		if len(val) >= 2 && val[0] == '"' && val[len(val)-1] == '"' {
			val = strings.ReplaceAll(val[1:len(val)-1], `\\`, `\`)
			val = strings.ReplaceAll(val, `\"`, `"`)
		}
		// IE6 sends the full client path.
		if key == "filename" && (len(val) >= 3 && val[1:3] == `:\` || strings.HasPrefix(val, `\\`)) {
			val = val[strings.LastIndex(val, `\`)+1:]
		}
		opts[key] = val
	}
	return segs[0], opts
}

// parseParam is a literal port of python-multipart's _parseparam (itself
// vendored from email.message), quirks included, operating on bytes.
func parseParam(in string) []string {
	s := ";" + in
	var out []string
	start := 0
	for pyFind(s, ";", start, len(s)) == start {
		start++
		end := pyFind(s, ";", start, len(s))
		ind, diff := start, 0
		for end > 0 {
			diff += pyCount(s, `"`, ind, end) - pyCount(s, `\"`, ind, end)
			if diff%2 == 0 {
				break
			}
			end, ind = ind, pyFind(s, ";", end+1, len(s))
		}
		if end < 0 {
			end = len(s)
		}
		var f string
		if i := pyFind(s, "=", start, end); i == -1 {
			f = pySlice(s, start, end)
		} else {
			f = strings.ToLower(strings.TrimRight(pySlice(s, start, i), " \t\n\r\x0b\x0c")) + "=" +
				strings.TrimLeft(pySlice(s, i+1, end), " \t\n\r\x0b\x0c")
		}
		out = append(out, strings.TrimSpace(f))
		start = end
	}
	return out
}

// pyIndices normalizes Python slice bounds.
func pyIndices(n, start, end int) (int, int) {
	if start < 0 {
		start = max(start+n, 0)
	}
	if end < 0 {
		end = max(end+n, 0)
	}
	return min(start, n), min(end, n)
}

func pySlice(s string, start, end int) string {
	start, end = pyIndices(len(s), start, end)
	if start >= end {
		return ""
	}
	return s[start:end]
}

func pyFind(s, sub string, start, end int) int {
	start, end = pyIndices(len(s), start, end)
	if start > end {
		return -1
	}
	i := strings.Index(s[start:end], sub)
	if i < 0 {
		return -1
	}
	return start + i
}

func pyCount(s, sub string, start, end int) int {
	start, end = pyIndices(len(s), start, end)
	if start > end {
		return 0
	}
	return strings.Count(s[start:end], sub)
}

// decodeCharset is Starlette's _user_safe_decode: the form charset, falling
// back to latin-1 when the bytes do not decode. Only UTF-8 and latin-1 are
// known here; other charsets are treated as UTF-8.
func decodeCharset(b []byte, charset string) string {
	switch strings.ToLower(strings.ReplaceAll(charset, "_", "-")) {
	case "latin-1", "latin1", "iso-8859-1", "iso8859-1", "l1":
		return latin1Decode(b)
	}
	if utf8.Valid(b) {
		return string(b)
	}
	return latin1Decode(b)
}

func isTokenChar(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' ||
		strings.IndexByte("!#$%&'*+-.^_`|~", c) >= 0
}

// multipartReader yields the parts of a multipart/form-data body one at a
// time; the data of each part streams through partReader.
type multipartReader struct {
	r       *bufio.Reader
	delim   []byte // "\r\n--" + boundary
	charset string
	started bool
	done    bool
	files   int
	fields  int
	cur     *partReader
}

type formPart struct {
	name     string
	filename string
	isFile   bool
	data     *partReader
}

func newMultipartReader(body io.Reader, boundary, charset string) (*multipartReader, error) {
	if len(boundary) > maxBoundaryLen {
		return nil, errBodyParse
	}
	return &multipartReader{
		r:       bufio.NewReaderSize(body, formBufferBytes),
		delim:   []byte("\r\n--" + boundary),
		charset: charset,
	}, nil
}

func (m *multipartReader) readByte() (byte, error) {
	c, err := m.r.ReadByte()
	if err == io.EOF {
		return 0, errTruncated
	}
	if err != nil {
		return 0, &bodyError{err}
	}
	return c, nil
}

// start consumes the opening boundary. It reports false when the form
// ended (or the body ended) before any part.
func (m *multipartReader) start() (bool, error) {
	c, err := m.readByte()
	if err != nil {
		return false, err
	}
	// python-multipart skips from a leading CR or LF to the next '-'.
	if c == '\r' || c == '\n' {
		for c != '-' {
			if c, err = m.readByte(); err != nil {
				return false, err
			}
		}
	}
	want := m.delim[2:]
	for i := 0; i < len(want); i++ {
		if i > 0 {
			if c, err = m.readByte(); err != nil {
				return false, err
			}
		}
		if c != want[i] {
			return false, errBodyParse
		}
	}
	return m.afterBoundary()
}

func (m *multipartReader) afterBoundary() (bool, error) {
	c, err := m.readByte()
	if err != nil {
		return false, err
	}
	switch c {
	case '-':
		if c, err = m.readByte(); err != nil {
			return false, err
		}
		if c != '-' {
			return false, errBodyParse
		}
		return false, nil
	case '\r':
		if c, err = m.readByte(); err != nil {
			return false, err
		}
		if c != '\n' {
			return false, errBodyParse
		}
		return true, nil
	}
	return false, errBodyParse
}

// next returns the next part, or io.EOF at the end of the form. A body
// that ends early is also a normal end, as in python-multipart. The
// previous part's data must have been read to completion.
func (m *multipartReader) next() (*formPart, error) {
	if m.done {
		return nil, io.EOF
	}
	if !m.started {
		m.started = true
		more, err := m.start()
		if err != nil || !more {
			return nil, m.finish(err)
		}
	} else if m.cur == nil || !m.cur.ended || m.cur.last {
		return nil, m.finish(nil)
	}
	disposition, err := m.readHeaders()
	if err != nil {
		return nil, m.finish(err)
	}
	_, opts := parseOptionsHeader(latin1Decode(disposition))
	name, ok := opts["name"]
	if !ok {
		return nil, m.finish(&formError{`The Content-Disposition header field "name" must be provided.`})
	}
	p := &formPart{name: decodeCharset(latin1Bytes(name), m.charset)}
	if fn, ok := opts["filename"]; ok {
		m.files++
		if m.files > maxFormFiles {
			return nil, m.finish(&formError{"Too many files. Maximum number of files is 1000."})
		}
		p.isFile = true
		p.filename = decodeCharset(latin1Bytes(fn), m.charset)
	} else {
		m.fields++
		if m.fields > maxFormFields {
			return nil, m.finish(&formError{"Too many fields. Maximum number of fields is 1000."})
		}
	}
	m.cur = &partReader{m: m}
	p.data = m.cur
	return p, nil
}

func (m *multipartReader) finish(err error) error {
	m.done = true
	if err == nil || errors.Is(err, errTruncated) {
		return io.EOF
	}
	return err
}

// latin1Bytes reverses latin1Decode.
func latin1Bytes(s string) []byte {
	b, _ := latin1Encode(s)
	return b
}

// readHeaders reads one part's header block and returns the raw value of
// the last Content-Disposition header.
func (m *multipartReader) readHeaders() ([]byte, error) {
	var disposition []byte
	for count := 0; ; {
		c, err := m.readByte()
		if err != nil {
			return nil, err
		}
		if c == '\r' {
			if c, err = m.readByte(); err != nil {
				return nil, err
			}
			if c != '\n' {
				return nil, errBodyParse
			}
			return disposition, nil
		}
		count++
		if count > maxHeaderCount {
			return nil, errBodyParse
		}
		var field []byte
		size := 0
		for c != ':' {
			if !isTokenChar(c) {
				return nil, errBodyParse
			}
			field = append(field, c)
			if size++; size > maxHeaderSize {
				return nil, errBodyParse
			}
			if c, err = m.readByte(); err != nil {
				return nil, err
			}
		}
		if len(field) == 0 {
			return nil, errBodyParse
		}
		size++ // the colon
		for {
			if c, err = m.readByte(); err != nil {
				return nil, err
			}
			if c != ' ' {
				break
			}
			if size++; size > maxHeaderSize {
				return nil, errBodyParse
			}
		}
		var value []byte
		for c != '\r' {
			value = append(value, c)
			if size++; size > maxHeaderSize {
				return nil, errBodyParse
			}
			if c, err = m.readByte(); err != nil {
				return nil, err
			}
		}
		if c, err = m.readByte(); err != nil {
			return nil, err
		}
		if c != '\n' {
			return nil, errBodyParse
		}
		if strings.EqualFold(string(field), "content-disposition") {
			disposition = value
		}
	}
}

// partReader streams one part's data up to the next delimiter that is
// followed by CRLF (another part) or "--" (end of form). It returns
// errTruncated if the body ends first and *bodyError on read failures.
type partReader struct {
	m     *multipartReader
	ended bool
	last  bool
	err   error
}

func (p *partReader) Read(b []byte) (int, error) {
	if p.ended {
		return 0, io.EOF
	}
	if p.err != nil {
		return 0, p.err
	}
	m := p.m
	need := len(m.delim) + 2
	for {
		buf, err := m.r.Peek(formBufferBytes)
		if err != nil && err != io.EOF && !errors.Is(err, bufio.ErrBufferFull) {
			p.err = &bodyError{err}
			if len(buf) == 0 {
				return 0, p.err
			}
		}
		eof := err == io.EOF
		i := bytes.Index(buf, m.delim)
		switch {
		case i > 0:
			n := copy(b, buf[:i])
			m.r.Discard(n)
			return n, nil
		case i == 0:
			if len(buf) < need {
				if eof {
					p.err = errTruncated
					return 0, p.err
				}
				// Peek blocks until the buffer is full or the body ends, so
				// a short buffer here always means EOF or an error.
				if p.err != nil {
					return 0, p.err
				}
				continue
			}
			switch string(buf[len(m.delim):need]) {
			case "\r\n":
				m.r.Discard(need)
				p.ended = true
				return 0, io.EOF
			case "--":
				m.r.Discard(need)
				p.ended, p.last = true, true
				return 0, io.EOF
			}
			// Not a real delimiter: its first byte is data.
			b[0] = buf[0]
			m.r.Discard(1)
			return 1, nil
		}
		// No delimiter in view: everything that cannot be the start of one
		// is data.
		safe := len(buf)
		if !eof {
			safe = max(len(buf)-len(m.delim)+1, 0)
		}
		if safe == 0 {
			if eof || p.err != nil {
				if p.err == nil {
					p.err = errTruncated
				}
				return 0, p.err
			}
			continue
		}
		n := copy(b, buf[:safe])
		m.r.Discard(n)
		return n, nil
	}
}

// urlencodedForm parses an application/x-www-form-urlencoded body like
// python-multipart's QuerystringParser with Starlette's limits.
func urlencodedForm(body io.Reader) ([]pyPair, error) {
	r := bufio.NewReaderSize(body, formBufferBytes)
	var out []pyPair
	var cur []byte
	hasEq := false
	flush := func() error {
		hasEq = false
		if len(cur) == 0 {
			return nil
		}
		name, value, _ := bytes.Cut(cur, []byte("="))
		cur = cur[:0]
		out = append(out, pyPair{
			name:  pyUnquote(strings.ReplaceAll(latin1Decode(name), "+", " ")),
			value: pyUnquote(strings.ReplaceAll(latin1Decode(value), "+", " ")),
		})
		if len(out) > maxFormFields {
			return &formError{"Too many fields. Maximum number of fields is 1000."}
		}
		return nil
	}
	for {
		c, err := r.ReadByte()
		if err == io.EOF {
			return out, flush()
		}
		if err != nil {
			return nil, &bodyError{err}
		}
		if c == '&' {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}
		cur = append(cur, c)
		size := len(cur)
		if hasEq || c == '=' {
			hasEq = true
			size--
		}
		if size > maxPartSize {
			return nil, &formError{"Field exceeded maximum size of 1024KB."}
		}
	}
}
