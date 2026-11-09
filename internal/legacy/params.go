package legacy

import (
	"math"
	"strconv"
	"strings"
	"unicode"
)

// fieldError is one entry of FastAPI's 422 "detail" list.
type fieldError struct {
	typ   string
	loc   [2]string
	msg   string
	input *string // nil renders as null
	// valueErrorCtx adds the "ctx":{"error":{}} pydantic attaches to
	// value_error entries.
	valueErrorCtx bool
}

const (
	msgBool    = "Input should be a valid boolean, unable to interpret input"
	msgInt     = "Input should be a valid integer, unable to parse string as an integer"
	msgMissing = "Field required"
	msgNotFile = "Value error, Expected UploadFile, received: <class 'str'>"
)

func renderValidationErrors(errs []fieldError) []byte {
	b := []byte(`{"detail":[`)
	for i, e := range errs {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, `{"type":`...)
		b = appendPyJSONString(b, e.typ)
		b = append(b, `,"loc":[`...)
		b = appendPyJSONString(b, e.loc[0])
		b = append(b, ',')
		b = appendPyJSONString(b, e.loc[1])
		b = append(b, `],"msg":`...)
		b = appendPyJSONString(b, e.msg)
		b = append(b, `,"input":`...)
		if e.input == nil {
			b = append(b, "null"...)
		} else {
			b = appendPyJSONString(b, *e.input)
		}
		if e.valueErrorCtx {
			b = append(b, `,"ctx":{"error":{}}`...)
		}
		b = append(b, '}')
	}
	return append(b, "]}"...)
}

// query holds parsed query parameters; Starlette's QueryParams.get, which
// FastAPI uses for scalar parameters, returns the last value of a repeated
// key.
type query map[string]string

func parseQuery(raw string) query {
	q := query{}
	for _, p := range parseQSL(raw) {
		q[p.name] = p.value
	}
	return q
}

// validator accumulates FastAPI query validation errors in declaration
// order, which is the order FastAPI reports them in.
type validator struct {
	q    query
	errs []fieldError
}

func (v *validator) str(name, def string) (string, bool) {
	s, ok := v.q[name]
	if !ok {
		return def, false
	}
	return s, true
}

func (v *validator) boolean(name string, def bool) bool {
	s, ok := v.q[name]
	if !ok {
		return def
	}
	b, ok := pyBool(s)
	if !ok {
		in := s
		v.errs = append(v.errs, fieldError{typ: "bool_parsing", loc: [2]string{"query", name}, msg: msgBool, input: &in})
	}
	return b
}

func (v *validator) integer(name string) *int {
	s, ok := v.q[name]
	if !ok {
		return nil
	}
	n, ok := pyInt(s)
	if !ok {
		in := s
		v.errs = append(v.errs, fieldError{typ: "int_parsing", loc: [2]string{"query", name}, msg: msgInt, input: &in})
		return nil
	}
	return &n
}

// pyBool implements pydantic's lax str -> bool: case-insensitive, no
// whitespace trimming.
func pyBool(s string) (bool, bool) {
	switch strings.ToLower(s) {
	case "1", "on", "t", "true", "y", "yes":
		return true, true
	case "0", "off", "f", "false", "n", "no":
		return false, true
	}
	return false, false
}

// pyInt implements pydantic's lax str -> int as observed on the reference:
// surrounding Unicode whitespace is ignored, one sign, ASCII digits with
// single underscores between them, and an optional all-zero fraction
// ("2.00"). Values beyond the int range saturate: pydantic accepts them as
// Python big ints, and no engine could use such a speaker count anyway.
func pyInt(s string) (int, bool) {
	s = strings.TrimFunc(s, unicode.IsSpace)
	if intPart, frac, found := strings.Cut(s, "."); found {
		if intPart == "" || frac == "" || strings.Trim(frac, "0") != "" {
			return 0, false
		}
		s = intPart
	}
	neg := false
	if s != "" && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		s = s[1:]
	}
	if s == "" {
		return 0, false
	}
	digits := make([]byte, 0, len(s))
	prevDigit := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= '0' && c <= '9':
			digits = append(digits, c)
			prevDigit = true
		case c == '_' && prevDigit && i+1 < len(s) && s[i+1] >= '0' && s[i+1] <= '9':
			prevDigit = false
		default:
			return 0, false
		}
	}
	n, err := strconv.ParseInt(string(digits), 10, 64)
	if err != nil || n > math.MaxInt {
		n = math.MaxInt
	}
	if neg {
		return -int(n), true
	}
	return int(n), true
}
