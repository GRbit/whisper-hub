package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/GRbit/whisper-hub/internal/audio"
	"github.com/GRbit/whisper-hub/internal/auth"
	"github.com/GRbit/whisper-hub/internal/config"
	"github.com/GRbit/whisper-hub/internal/hub"
	"github.com/GRbit/whisper-hub/internal/legacy"
	"github.com/GRbit/whisper-hub/internal/openaiapi"
	"github.com/GRbit/whisper-hub/internal/workerproc"
	"github.com/GRbit/whisper-hub/internal/wyoming"
)

// gateway is the composed service with its listeners bound.
type gateway struct {
	cfg  config.Config
	log  *slog.Logger
	auth *auth.Auth
	hub  *hub.Hub

	servers []*http.Server
	httpLns []net.Listener
	wy      *wyoming.Server
	wyLn    net.Listener
	worker  *workerproc.Proc
}

// newGateway builds every component and binds the listeners, so a port
// conflict or a missing ffmpeg fails startup instead of the first request.
func newGateway(ctx context.Context, cfg config.Config, log *slog.Logger) (_ *gateway, err error) {
	g := &gateway{cfg: cfg, log: log}
	defer func() {
		if err != nil {
			g.closeAll()
		}
	}()

	// Before anything logs "started": a wrong ASR_ENGINE is a deployment
	// error the operator must see first.
	if _, err := cfg.Worker.Env(os.Environ()); err != nil {
		return nil, err
	}
	dec := audio.New(cfg.Audio, log)
	if err := dec.Check(ctx); err != nil {
		return nil, fmt.Errorf("%w (install ffmpeg or set audio.ffmpeg_path)", err)
	}
	// Only /v1/ has its own rejection dialect; legacy paths get auth's
	// default, the byte-identical 401 of the old sidecar.
	if g.auth, err = auth.New(cfg.Auth, map[string]func(http.ResponseWriter, error){"/v1/": openaiapi.WriteError}); err != nil {
		return nil, err
	}
	if g.hub, err = hub.New(cfg.Hub, g.auth, g.auth, log); err != nil {
		return nil, err
	}

	mux := http.NewServeMux()
	// legacy owns "/" (its JSON 404 catch-all); the more specific patterns
	// of the others win over it.
	legacy.New(cfg.Legacy, g.hub, dec).Register(mux)
	openaiapi.New(cfg.OpenAI, g.hub, dec, log).Register(mux)
	mux.Handle("GET /healthz", g.hub.HealthHandler())
	mux.Handle(auth.UsagePath, g.auth.UsageHandler())

	if err := g.listenHTTP(cfg.HTTP.Listen, g.auth.Middleware(mux)); err != nil {
		return nil, err
	}
	if cfg.HTTP.OpenListen != "" {
		// Deliberately without the auth middleware: this is the old open
		// webservice port of the bundled sidecar setup.
		if err := g.listenHTTP(cfg.HTTP.OpenListen, mux); err != nil {
			return nil, err
		}
	}
	if cfg.Wyoming.Enabled {
		wc := cfg.Wyoming.Config
		if wc.Listen == "" {
			wc.Listen = ":10300"
		}
		if wc.Version == "" {
			wc.Version = version
		}
		if g.wy, err = wyoming.New(wc, g.hub, dec, g.auth.WyomingIdentity, log); err != nil {
			return nil, err
		}
		if g.wyLn, err = net.Listen("tcp", wc.Listen); err != nil {
			return nil, fmt.Errorf("wyoming listen: %w", err)
		}
	}

	// Last, so a bad port or config does not leave a worker loading a model.
	if g.worker, err = workerproc.Start(cfg.Worker, os.Environ(), log); err != nil {
		return nil, err
	}

	workers := make([]string, 0, len(cfg.Hub.Workers))
	for _, w := range cfg.Hub.Workers {
		workers = append(workers, w.URL)
	}
	authState := "off"
	if cfg.Auth.UsersFile != "" {
		authState = "on (" + cfg.Auth.UsersFile + ")"
	}
	log.Info("whisper-hub started", "version", version, "http", g.addr(0), "open_http", g.openAddr(),
		"wyoming", g.wyAddr(), "workers", workers, "auth", authState, "config", cfg.File)
	return g, nil
}

func (g *gateway) listenHTTP(addr string, h http.Handler) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("http listen: %w", err)
	}
	g.httpLns = append(g.httpLns, ln)
	// No read or write timeout: uploads and transcriptions of long files
	// legitimately take many minutes. Slowloris is bounded by the header one.
	g.servers = append(g.servers, &http.Server{Handler: h, ReadHeaderTimeout: 30 * time.Second,
		IdleTimeout: 2 * time.Minute, ErrorLog: slog.NewLogLogger(g.log.Handler(), slog.LevelWarn)})
	return nil
}

func (g *gateway) addr(i int) string {
	if i < len(g.httpLns) {
		return g.httpLns[i].Addr().String()
	}
	return ""
}

func (g *gateway) openAddr() string { return g.addr(1) }

func (g *gateway) wyAddr() string {
	if g.wyLn == nil {
		return ""
	}
	return g.wyLn.Addr().String()
}

// Run serves until ctx is canceled or a listener fails, then shuts down:
// stop accepting, let running requests finish within the grace period,
// then close the hub and auth (which writes the final usage snapshot).
func (g *gateway) Run(ctx context.Context) error {
	authCtx, stopAuth := context.WithCancel(context.Background())
	authDone := make(chan struct{})
	go func() { defer close(authDone); g.auth.Run(authCtx) }()

	fatal := make(chan error, len(g.servers)+1)
	for i, srv := range g.servers {
		go func() {
			if err := srv.Serve(g.httpLns[i]); !errors.Is(err, http.ErrServerClosed) {
				fatal <- fmt.Errorf("http serve: %w", err)
			}
		}()
	}
	wyCtx, stopWy := context.WithCancel(context.Background())
	var wyDone chan error // nil (never ready) while Wyoming is off
	if g.wy != nil {
		wyDone = make(chan error, 1)
		go func() { wyDone <- g.wy.Serve(wyCtx, g.wyLn) }()
	}

	var workerDone <-chan struct{} // nil (never ready) without a managed worker
	if g.worker != nil {
		workerDone = g.worker.Done()
	}

	var runErr error
	wyStopped := false
	select {
	case <-ctx.Done():
	case runErr = <-fatal:
	case runErr = <-wyDone:
		wyStopped = true
		if runErr == nil {
			runErr = errors.New("wyoming server stopped")
		}
	case <-workerDone:
		// Exit so the container restarts, rather than answer "unavailable"
		// forever.
		runErr = fmt.Errorf("engine worker exited: %v", g.worker.Err())
	}
	grace := g.cfg.HTTP.ShutdownGrace
	g.log.Info("shutting down", "grace", grace, "err", runErr)

	stopWy()
	shCtx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	var wg sync.WaitGroup
	for _, srv := range g.servers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := srv.Shutdown(shCtx); err != nil {
				// Grace is over: cancel what still runs, which also
				// cancels the worker jobs behind those requests.
				g.log.Warn("grace period over, cutting off running requests", "err", err)
				_ = srv.Close()
			}
		}()
	}
	wg.Wait()
	g.log.Debug("shutdown: http servers drained")
	if wyDone != nil && !wyStopped {
		if err := <-wyDone; err != nil && runErr == nil {
			runErr = err
		}
	}
	g.log.Debug("shutdown: wyoming stopped")
	// The worker goes last: requests still draining above need it.
	if g.worker != nil {
		g.worker.Stop()
	}
	stopAuth()
	<-authDone
	g.log.Debug("shutdown: auth stopped")
	g.closeAll()
	g.log.Info("stopped")
	return runErr
}

func (g *gateway) closeAll() {
	if g.worker != nil {
		g.worker.Stop()
	}
	for _, ln := range g.httpLns {
		_ = ln.Close()
	}
	if g.wyLn != nil {
		_ = g.wyLn.Close()
	}
	if g.hub != nil {
		g.hub.Close()
	}
	if g.auth != nil {
		if err := g.auth.Close(); err != nil {
			g.log.Error("closing auth", "err", err)
		}
	}
}
