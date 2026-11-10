package core

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// NetworkAdminUser is the user name recorded for network admin access, see
// Identity.NetworkAdmin.
const NetworkAdminUser = "admin"

// WyomingUser is the user every Wyoming connection acts as. Wyoming has no
// credentials, so only the network restricts who connects.
const WyomingUser = "wyoming"

// Identity is the caller of a request.
type Identity struct {
	// User is the account name; NetworkAdminUser for network admin access.
	User string
	// NetworkAdmin is set when no credentials were checked: authentication
	// is off, the path is not protected, or the request came through the
	// unauthenticated listener. The network restricts who gets that far,
	// so this is admin access. It is not the users file admin flag.
	NetworkAdmin bool
	// Models holds the model id patterns this caller may use, as written
	// in the users file. Interpreting them is the Authorizer's job.
	Models []string
}

type identityKey struct{}

var networkAdmin = &Identity{User: NetworkAdminUser, NetworkAdmin: true}

// WithIdentity returns a context carrying id.
func WithIdentity(ctx context.Context, id *Identity) context.Context {
	return context.WithValue(ctx, identityKey{}, id)
}

// IdentityFrom returns the caller stored in ctx. It never returns nil: a
// context without an identity yields network admin access.
func IdentityFrom(ctx context.Context) *Identity {
	if id, ok := ctx.Value(identityKey{}).(*Identity); ok && id != nil {
		return id
	}
	return networkAdmin
}

// Authorizer decides which models a caller may use.
type Authorizer interface {
	// AllowModel reports whether id may run the model with this resolved
	// id. isDefault is true for the hub's default model.
	AllowModel(id *Identity, model string, isDefault bool) bool
}

// UsageEvent describes one finished job, successful or not.
type UsageEvent struct {
	Time         time.Time
	User         string
	Protocol     string
	Endpoint     string
	Model        string
	Engine       string
	Task         Task
	AudioSeconds float64
	Queued       time.Duration // time spent waiting for the worker
	Elapsed      time.Duration // total time including Queued
	// Status is "ok" or the ErrorKind name of the failure.
	Status string
	Error  string
}

// UsageRecorder persists per-user accounting. Record must not block the
// request path for long and must be safe for concurrent use.
type UsageRecorder interface {
	Record(ctx context.Context, ev UsageEvent)
}

// ErrorKind classifies failures so each protocol can map them to its own
// status codes and error shapes.
type ErrorKind int

// Error kinds.
const (
	KindInternal        ErrorKind = iota
	KindBadRequest                // invalid parameters
	KindBadAudio                  // upload could not be decoded to audio
	KindUnauthenticated           // missing or wrong credentials
	KindForbidden                 // authenticated, but the model is not allowed
	KindModelNotFound             // no such model or alias
	KindUnavailable               // no healthy engine worker, or queue full
	KindCanceled                  // client went away
)

// String returns the snake_case name used in logs and usage records.
func (k ErrorKind) String() string {
	switch k {
	case KindBadRequest:
		return "bad_request"
	case KindBadAudio:
		return "bad_audio"
	case KindUnauthenticated:
		return "unauthenticated"
	case KindForbidden:
		return "forbidden"
	case KindModelNotFound:
		return "model_not_found"
	case KindUnavailable:
		return "unavailable"
	case KindCanceled:
		return "canceled"
	default:
		return "internal"
	}
}

// Error is the error type returned across core interfaces.
type Error struct {
	Kind ErrorKind
	// Message is safe to show to clients: no server paths, no subprocess
	// output. Diagnostics go to the log and to Err.
	Message string
	Err     error
}

// Error implements the error interface.
func (e *Error) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s: %v", e.Kind, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

// Unwrap returns the underlying cause.
func (e *Error) Unwrap() error { return e.Err }

// Errorf builds an *Error with a client-safe message.
func Errorf(kind ErrorKind, format string, args ...any) *Error {
	return &Error{Kind: kind, Message: fmt.Sprintf(format, args...)}
}

// WrapError builds an *Error that keeps cause for the logs.
func WrapError(kind ErrorKind, cause error, format string, args ...any) *Error {
	return &Error{Kind: kind, Message: fmt.Sprintf(format, args...), Err: cause}
}

// KindOf returns the kind of err, mapping context cancellation to
// KindCanceled and anything unclassified to KindInternal.
func KindOf(err error) ErrorKind {
	var e *Error
	if errors.As(err, &e) {
		return e.Kind
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return KindCanceled
	}
	return KindInternal
}

// ClientMessage returns the text that may be shown to a client for err.
func ClientMessage(err error) string {
	var e *Error
	if errors.As(err, &e) {
		return e.Message
	}
	return "internal error"
}
