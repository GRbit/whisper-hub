// Package wyoming serves the speech-to-text side of the Wyoming protocol,
// as used by Home Assistant voice pipelines, on top of core.Service.
package wyoming

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

// IdentifyFunc maps a connection's remote address to the caller. Wyoming has
// no credentials, so this is the only authentication hook; an error rejects
// the connection.
type IdentifyFunc func(remote net.Addr) (*core.Identity, error)

// Server is a Wyoming speech-to-text server.
type Server struct {
	cfg      Config
	svc      core.Service
	dec      core.AudioDecoder
	identify IdentifyFunc
	log      *slog.Logger
	limits   limits
}

// New returns a Server. identify may be nil, which gives every connection
// network admin access (core.Identity.NetworkAdmin); logger may be nil for
// slog.Default().
func New(cfg Config, svc core.Service, dec core.AudioDecoder, identify IdentifyFunc, logger *slog.Logger) (*Server, error) {
	if svc == nil || dec == nil {
		return nil, errors.New("wyoming: Service and AudioDecoder are required")
	}
	cfg, err := cfg.withDefaults()
	if err != nil {
		return nil, err
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{
		cfg:      cfg,
		svc:      svc,
		dec:      dec,
		identify: identify,
		log:      logger.With("component", "wyoming"),
		limits: limits{
			MaxHeaderBytes:  cfg.MaxHeaderBytes,
			MaxDataBytes:    cfg.MaxDataBytes,
			MaxPayloadBytes: cfg.MaxPayloadBytes,
		},
	}, nil
}

// ListenAndServe listens on Config.Listen and calls Serve.
func (s *Server) ListenAndServe(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.cfg.Listen)
	if err != nil {
		return fmt.Errorf("wyoming: listen: %w", err)
	}
	return s.Serve(ctx, ln)
}

// Serve accepts connections on ln until ctx is canceled, then shuts down
// gracefully: idle connections close at once, utterances still being
// recorded get an error, and transcriptions already running may finish for
// up to Config.ShutdownGrace. It closes ln and returns nil after a graceful
// shutdown, or the accept error that stopped it.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	s.log.Info("wyoming server listening", "addr", ln.Addr().String(),
		"streaming", s.cfg.StreamTranscripts, "zeroconf", s.cfg.Zeroconf.Enabled)

	// hard outlives ctx by the shutdown grace and is what jobs run under.
	hard, hardCancel := context.WithCancel(context.WithoutCancel(ctx))
	defer hardCancel()

	if s.cfg.Zeroconf.Enabled {
		stop, err := s.startZeroconf(ctx, ln.Addr())
		if err != nil {
			// Discovery is a convenience; manual setup in HA still works.
			s.log.Warn("wyoming zeroconf announcement failed", "err", err)
		} else {
			defer stop()
		}
	}

	stopWatch := make(chan struct{})
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		select {
		case <-ctx.Done():
		case <-stopWatch:
			return
		}
		s.log.Info("wyoming server shutting down", "grace", s.cfg.ShutdownGrace)
		_ = ln.Close()
		if !enabled(s.cfg.ShutdownGrace) {
			hardCancel()
			return
		}
		t := time.NewTimer(s.cfg.ShutdownGrace)
		defer t.Stop()
		select {
		case <-t.C:
			hardCancel()
		case <-stopWatch:
		}
	}()

	var wg sync.WaitGroup
	var active atomic.Int64
	var acceptErr error
	backoff := 5 * time.Millisecond
	for {
		nc, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			if errors.Is(err, net.ErrClosed) {
				acceptErr = err
				break
			}
			// Same policy as net/http: transient errors such as EMFILE must
			// not kill the server.
			s.log.Warn("wyoming accept failed, retrying", "err", err, "backoff", backoff)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
			}
			backoff = min(backoff*2, time.Second)
			continue
		}
		backoff = 5 * time.Millisecond
		if s.cfg.MaxConnections > 0 && active.Load() >= int64(s.cfg.MaxConnections) {
			s.log.Warn("wyoming connection refused: too many connections",
				"remote", nc.RemoteAddr().String(), "max", s.cfg.MaxConnections)
			wg.Add(1)
			go func() {
				defer wg.Done()
				c := &conn{s: s, nc: nc, w: newFrameWriter(nc), log: s.log}
				c.sendError(core.KindUnavailable.String(), "too many connections")
				_ = nc.Close()
			}()
			continue
		}
		active.Add(1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer active.Add(-1)
			s.handleConn(ctx, hard, nc)
		}()
	}

	if acceptErr != nil {
		// Not a requested shutdown: treat it like one for the live connections.
		hardCancel()
	}
	wg.Wait()
	close(stopWatch)
	<-watchDone
	_ = ln.Close()
	if acceptErr != nil {
		return fmt.Errorf("wyoming: accept: %w", acceptErr)
	}
	s.log.Info("wyoming server stopped")
	return nil
}

func (s *Server) handleConn(shutdown, hard context.Context, nc net.Conn) {
	remote := nc.RemoteAddr()
	log := s.log.With("remote", remote.String())

	id := core.IdentityFrom(context.Background())
	if s.identify != nil {
		got, err := s.identify(remote)
		if err != nil {
			log.Warn("wyoming connection rejected", "kind", core.KindOf(err).String(), "err", err)
			c := &conn{s: s, nc: nc, w: newFrameWriter(nc), log: log}
			c.sendError(core.KindOf(err).String(), core.ClientMessage(err))
			lingerClose(nc)
			return
		}
		if got != nil {
			id = got
		}
	}
	log = log.With("user", id.User)

	ctx, cancel := context.WithCancel(core.WithIdentity(hard, id))
	c := &conn{
		s:      s,
		nc:     nc,
		w:      newFrameWriter(nc),
		log:    log,
		ctx:    ctx,
		cancel: cancel,
		hard:   hard,
		jobRes: make(chan bool, 1),
	}
	log.Debug("wyoming connection opened")
	reason := c.run(shutdown)
	cancel()
	c.close()
	c.jobs.Wait()
	log.Debug("wyoming connection closed", "reason", reason)
}

// lingerClose closes a rejected connection without losing the error event
// just sent. Clients such as Home Assistant write their whole utterance
// before reading; closing with that data unread makes the kernel answer
// with RST, which fails the client's writes and discards the error event
// in its receive buffer. Half-closing and draining for a bounded time turns
// that into an orderly FIN.
func lingerClose(nc net.Conn) {
	if tc, ok := nc.(*net.TCPConn); ok {
		_ = tc.CloseWrite()
		_ = nc.SetReadDeadline(time.Now().Add(2 * time.Second))
		_, _ = io.CopyN(io.Discard, nc, 16<<20)
	}
	_ = nc.Close()
}
