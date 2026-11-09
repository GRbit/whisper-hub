package openaiapi

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Error type values used by the OpenAI API.
const (
	typeInvalidRequest = "invalid_request_error"
	typeServer         = "server_error"
)

// apiError is an error already shaped for the OpenAI wire format.
type apiError struct {
	Status  int     `json:"-"`
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

func (e *apiError) Error() string { return e.Message }

type errorBody struct {
	Error *apiError `json:"error"`
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func badRequest(param, code, format string, args ...any) *apiError {
	return &apiError{
		Status:  http.StatusBadRequest,
		Message: fmt.Sprintf(format, args...),
		Type:    typeInvalidRequest,
		Param:   strPtr(param),
		Code:    strPtr(code),
	}
}

func invalidValue(param, value string, supported string) *apiError {
	return badRequest(param, "invalid_value",
		"Invalid value: '%s'. Supported values are: %s.", value, supported)
}

// toAPIError maps any error to the OpenAI shape. Only client-safe text from
// core.Error.Message is exposed; the cause stays in the logs.
func toAPIError(err error) *apiError {
	if ae, ok := err.(*apiError); ok {
		return ae
	}
	msg := core.ClientMessage(err)
	switch core.KindOf(err) {
	case core.KindBadRequest:
		return &apiError{Status: http.StatusBadRequest, Message: msg, Type: typeInvalidRequest}
	case core.KindBadAudio:
		return &apiError{Status: http.StatusBadRequest, Message: msg, Type: typeInvalidRequest,
			Param: strPtr("file"), Code: strPtr("invalid_value")}
	case core.KindUnauthenticated:
		return &apiError{Status: http.StatusUnauthorized, Message: msg, Type: typeInvalidRequest,
			Code: strPtr("invalid_api_key")}
	case core.KindForbidden:
		return &apiError{Status: http.StatusForbidden, Message: msg, Type: typeInvalidRequest,
			Param: strPtr("model"), Code: strPtr("model_not_allowed")}
	case core.KindModelNotFound:
		return &apiError{Status: http.StatusNotFound, Message: msg, Type: typeInvalidRequest,
			Param: strPtr("model"), Code: strPtr("model_not_found")}
	case core.KindUnavailable:
		return &apiError{Status: http.StatusServiceUnavailable, Message: msg, Type: typeServer}
	case core.KindCanceled:
		// Nginx convention for "client closed request"; the client is gone and
		// will not read it, but logs and usage records stay distinguishable.
		return &apiError{Status: 499, Message: "request canceled", Type: typeInvalidRequest}
	default:
		return &apiError{Status: http.StatusInternalServerError,
			Message: "The server had an error while processing your request. Sorry about that!",
			Type:    typeServer}
	}
}

// WriteError writes err as an OpenAI error response with the matching
// status code.
//
// It is exported so that middleware in front of /v1 (authentication) can
// answer in the shape OpenAI SDKs parse into typed exceptions.
func WriteError(w http.ResponseWriter, err error) {
	ae := toAPIError(err)
	writeJSON(w, ae.Status, errorBody{Error: ae})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		// Only our own response structs get here; they always marshal.
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}
