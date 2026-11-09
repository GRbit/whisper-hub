// Command whisper-hub is the speech-to-text gateway: legacy
// whisper-asr-webservice API, OpenAI-compatible API and Wyoming in front of
// Python engine workers.
//
//	whisper-hub [serve] [--config file] [--host h] [--port p]
//	whisper-hub hash      print a bcrypt hash for a users file password
//	whisper-hub apikey    print a new API key and its users file hash
//	whisper-hub config    print the effective configuration
//	whisper-hub healthcheck  exit 0 when /healthz of the local gateway is ok
//	whisper-hub version
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/GRbit/whisper-hub/internal/config"
)

// version is set with -ldflags "-X main.version=...".
var version = "dev"

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin *os.File, stdout, stderr io.Writer) int {
	cmd := "serve"
	if len(args) > 0 && len(args[0]) > 0 && args[0][0] != '-' {
		cmd, args = args[0], args[1:]
	}
	var err error
	switch cmd {
	// whisper-asr-webservice: the upstream image's command, for deployments
	// that spell it out.
	case "serve", "whisper-asr-webservice":
		err = serve(args, stderr)
	case "healthcheck":
		err = healthcheck(args, stderr)
	case "config":
		var cfg config.Config
		if cfg, err = config.Load(args, os.LookupEnv, stderr); err == nil {
			var b []byte
			if b, err = cfg.YAML(); err == nil {
				_, err = stdout.Write(b)
			}
		}
	case "hash":
		err = hashCmd(stdin, stdout, stderr)
	case "apikey":
		err = apikeyCmd(stdout)
	case "version":
		_, err = fmt.Fprintln(stdout, version)
	default:
		err = fmt.Errorf("unknown command %q (serve, hash, apikey, config, healthcheck, version)", cmd)
	}
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		fmt.Fprintln(stderr, "whisper-hub:", err)
		return 1
	}
	return 0
}

func serve(args []string, stderr io.Writer) error {
	cfg, err := config.Load(args, os.LookupEnv, stderr)
	if err != nil {
		return err
	}
	log := newLogger(cfg.Log, stderr)
	// auth logs through the default logger.
	slog.SetDefault(log)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	g, err := newGateway(ctx, cfg, log)
	if err != nil {
		return err
	}
	hup := make(chan os.Signal, 1)
	signal.Notify(hup, syscall.SIGHUP)
	defer signal.Stop(hup)
	go func() {
		for range hup {
			log.Info("SIGHUP: reloading users file")
			if err := g.auth.Reload(); err != nil {
				log.Error("reload failed, previous users stay in force", "err", err)
			}
		}
	}()
	return g.Run(ctx)
}

func newLogger(c config.Log, w io.Writer) *slog.Logger {
	var level slog.Level
	_ = level.UnmarshalText([]byte(c.Level)) // validated by config.Load
	opts := &slog.HandlerOptions{Level: level}
	if c.Format == "json" {
		return slog.New(slog.NewJSONHandler(w, opts))
	}
	return slog.New(slog.NewTextHandler(w, opts))
}

// healthcheck asks the gateway configured like this process (same flags
// and environment) for /healthz, for container HEALTHCHECKs in images
// without curl or wget.
func healthcheck(args []string, stderr io.Writer) error {
	cfg, err := config.Load(args, os.LookupEnv, stderr)
	if err != nil {
		return err
	}
	_, port, err := net.SplitHostPort(cfg.HTTP.Listen)
	if err != nil {
		return err
	}
	c := &http.Client{Timeout: 5 * time.Second}
	resp, err := c.Get("http://" + net.JoinHostPort("127.0.0.1", port) + "/healthz")
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("healthz: %s", resp.Status)
	}
	return nil
}
