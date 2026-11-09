package legacy

import (
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// This file reproduces the few Python stdlib behaviors that leak into the
// reference service's responses: json.dumps(ensure_ascii=False) strings,
// float repr, urllib.parse.quote/unquote and parse_qsl, latin-1 header
// decoding and UTF-8 decoding with errors="replace".

// appendPyJSONString appends s as json.dumps(s, ensure_ascii=False) would.
func appendPyJSONString(b []byte, s string) []byte {
	const hex = "0123456789abcdef"
	b = append(b, '"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b = append(b, '\\', '"')
		case '\\':
			b = append(b, '\\', '\\')
		case '\n':
			b = append(b, '\\', 'n')
		case '\r':
			b = append(b, '\\', 'r')
		case '\t':
			b = append(b, '\\', 't')
		case '\b':
			b = append(b, '\\', 'b')
		case '\f':
			b = append(b, '\\', 'f')
		default:
			if c < 0x20 {
				b = append(b, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xf])
			} else {
				b = append(b, c)
			}
		}
	}
	return append(b, '"')
}

// pyFloatRepr formats f as Python's repr(float) does ("1.0", "1e-05",
// "1e+16"), which is also what json.dumps emits. ok is false for NaN and
// infinities, which Starlette's JSONResponse refuses to serialize.
func pyFloatRepr(f float64) (s string, ok bool) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "", false
	}
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0", true
		}
		return "0.0", true
	}
	e := strconv.FormatFloat(f, 'e', -1, 64)
	exp, _ := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	// Python switches to exponent notation when the decimal point position
	// (exp+1) is <= -4 or > 16.
	if exp < -4 || exp >= 16 {
		return e, true
	}
	s = strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsRune(s, '.') {
		s += ".0"
	}
	return s, true
}

// pyQuote is urllib.parse.quote(s) with the default safe="/".
func pyQuote(s string) string {
	const hex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' ||
			c == '_' || c == '.' || c == '-' || c == '~' || c == '/' {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(hex[c>>4])
		b.WriteByte(hex[c&0xf])
	}
	return b.String()
}

func unhex(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

// pyUnquote is urllib.parse.unquote(s, errors="replace") on a str (Go
// string holding valid UTF-8): runs of ASCII get %XX decoded and the result
// decoded as UTF-8 with replacement; invalid escapes stay literal.
func pyUnquote(s string) string {
	if !strings.Contains(s, "%") {
		return s
	}
	var out strings.Builder
	var run []byte
	flush := func() {
		if len(run) > 0 {
			out.WriteString(decodeUTF8Replace(run))
			run = run[:0]
		}
	}
	for i := 0; i < len(s); {
		c := s[i]
		if c >= utf8.RuneSelf {
			flush()
			_, n := utf8.DecodeRuneInString(s[i:])
			out.WriteString(s[i : i+n])
			i += n
			continue
		}
		if c == '%' && i+2 < len(s) {
			if hi, ok := unhex(s[i+1]); ok {
				if lo, ok := unhex(s[i+2]); ok {
					run = append(run, hi<<4|lo)
					i += 3
					continue
				}
			}
		}
		run = append(run, c)
		i++
	}
	flush()
	return out.String()
}

// decodeUTF8Replace decodes b like Python's bytes.decode("utf-8",
// "replace"): every maximal invalid subpart becomes one U+FFFD, which is
// not what strings.ToValidUTF8 does (it merges whole runs).
func decodeUTF8Replace(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var out strings.Builder
	for i := 0; i < len(b); {
		r, n := utf8.DecodeRune(b[i:])
		if r != utf8.RuneError || n > 1 {
			out.WriteRune(r)
			i += n
			continue
		}
		out.WriteRune(utf8.RuneError)
		i += invalidSubpartLen(b[i:])
	}
	return out.String()
}

// invalidSubpartLen returns the length of the maximal prefix of b that
// starts a well-formed UTF-8 sequence but is cut short (at least 1).
func invalidSubpartLen(b []byte) int {
	c := b[0]
	var need int
	lo, hi := byte(0x80), byte(0xBF)
	switch {
	case c >= 0xC2 && c <= 0xDF:
		need = 1
	case c == 0xE0:
		need, lo = 2, 0xA0
	case c >= 0xE1 && c <= 0xEC, c == 0xEE, c == 0xEF:
		need = 2
	case c == 0xED:
		need, hi = 2, 0x9F
	case c == 0xF0:
		need, lo = 3, 0x90
	case c >= 0xF1 && c <= 0xF3:
		need = 3
	case c == 0xF4:
		need, hi = 3, 0x8F
	default:
		return 1
	}
	n := 1
	for ; n <= need && n < len(b); n++ {
		if b[n] < lo || b[n] > hi {
			break
		}
		lo, hi = 0x80, 0xBF
	}
	return n
}

// latin1Decode maps every byte to the code point of the same value, as
// Python's bytes.decode("latin-1").
func latin1Decode(b []byte) string {
	ascii := true
	for _, c := range b {
		if c >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	if ascii {
		return string(b)
	}
	var out strings.Builder
	for _, c := range b {
		out.WriteRune(rune(c))
	}
	return out.String()
}

// latin1Encode is str.encode("latin-1"); ok is false when s has a code
// point above U+00FF.
func latin1Encode(s string) (b []byte, ok bool) {
	b = make([]byte, 0, len(s))
	for _, r := range s {
		if r > 0xFF {
			return nil, false
		}
		b = append(b, byte(r))
	}
	return b, true
}

type pyPair struct{ name, value string }

// parseQSL is Starlette's query parsing: the raw query bytes decoded as
// latin-1, then urllib.parse.parse_qsl(keep_blank_values=True). Unlike
// url.ParseQuery it never drops a pair over a bad escape or a ';'.
func parseQSL(raw string) []pyPair {
	s := latin1Decode([]byte(raw))
	var out []pyPair
	for _, part := range strings.Split(s, "&") {
		if part == "" {
			continue
		}
		name, value, _ := strings.Cut(part, "=")
		out = append(out, pyPair{
			name:  pyUnquote(strings.ReplaceAll(name, "+", " ")),
			value: pyUnquote(strings.ReplaceAll(value, "+", " ")),
		})
	}
	return out
}
