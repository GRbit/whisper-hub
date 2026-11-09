// Package workerproc runs the engine worker as a child of the gateway, for
// the all-in-one image: one PID 1 that owns the worker, stops it last on
// shutdown, and exits when it dies, so the container is restarted instead
// of serving "unavailable" forever.
package workerproc

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"slices"
	"strings"
	"syscall"
	"time"
)

// Config describes the worker process. The zero value starts nothing: the
// gateway then talks to workers started elsewhere.
type Config struct {
	// Command is the worker command line, e.g. ["python", "-m",
	// "whisper_worker"]. Default empty: no managed worker.
	Command []string `yaml:"command"`
	// Engine is the engine installed next to Command. When set, ASR_ENGINE
	// must be unset (it is then set to Engine) or equal to it. Default "".
	Engine string `yaml:"engine"`
	// ImageTag is how the error for a missing engine names the image to
	// use; "%s" is replaced by the requested engine. Default
	// "whisper-hub:%s".
	ImageTag string `yaml:"image_tag"`
	// StopTimeout is how long the worker may take to exit after SIGTERM
	// before it is killed. Default 10s.
	StopTimeout time.Duration `yaml:"stop_timeout"`
}

// Engines are the ASR_ENGINE values the worker knows.
var Engines = []string{"faster_whisper", "whisperx", "openai_whisper", "fake"}

// Proc is a running worker process.
type Proc struct {
	cmd     *exec.Cmd
	log     *slog.Logger
	timeout time.Duration
	done    chan struct{}
	err     error
}

// Env returns env adjusted for the worker: ASR_ENGINE defaults to the
// installed engine instead of upstream's openai_whisper, and asking for an
// engine the image lacks is an error naming the image that has it.
func (c Config) Env(env []string) ([]string, error) {
	requested := ""
	for _, kv := range env {
		if v, ok := strings.CutPrefix(kv, "ASR_ENGINE="); ok {
			requested = strings.TrimSpace(v)
		}
	}
	if c.Engine == "" {
		return env, nil
	}
	if requested == "" {
		return append(slices.Clone(env), "ASR_ENGINE="+c.Engine), nil
	}
	if requested == c.Engine {
		return env, nil
	}
	tag := c.ImageTag
	if tag == "" {
		tag = "whisper-hub:%s"
	}
	if !slices.Contains(Engines, requested) {
		return nil, fmt.Errorf("ASR_ENGINE=%q is not an engine (%s)", requested, strings.Join(Engines, ", "))
	}
	return nil, fmt.Errorf("ASR_ENGINE=%s, but this image contains the %s engine only: use the %s image, or unset ASR_ENGINE",
		requested, c.Engine, strings.ReplaceAll(tag, "%s", requested))
}

// Start launches the worker with env (normally os.Environ()). It returns a
// nil Proc and no error when no command is configured.
func Start(c Config, env []string, log *slog.Logger) (*Proc, error) {
	if len(c.Command) == 0 {
		return nil, nil
	}
	if log == nil {
		log = slog.Default()
	}
	env, err := c.Env(env)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(c.Command[0], c.Command[1:]...)
	cmd.Env = env
	// The worker logs straight to our stderr, already formatted.
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	setProcAttrs(cmd)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start engine worker: %w", err)
	}
	p := &Proc{cmd: cmd, log: log, timeout: c.StopTimeout, done: make(chan struct{})}
	if p.timeout <= 0 {
		p.timeout = 10 * time.Second
	}
	log.Info("engine worker started", "pid", cmd.Process.Pid, "command", c.Command)
	go func() {
		p.err = cmd.Wait()
		close(p.done)
	}()
	return p, nil
}

// Done is closed when the worker has exited; Err then tells how.
func (p *Proc) Done() <-chan struct{} { return p.done }

// Err is the exit error once Done is closed (nil for a zero exit).
func (p *Proc) Err() error {
	<-p.done
	return p.err
}

// Stop sends SIGTERM, waits up to StopTimeout, then kills the worker.
func (p *Proc) Stop() {
	select {
	case <-p.done:
		return
	default:
	}
	p.log.Info("stopping engine worker", "pid", p.cmd.Process.Pid, "timeout", p.timeout)
	if err := p.cmd.Process.Signal(syscall.SIGTERM); err != nil && !errors.Is(err, os.ErrProcessDone) {
		p.log.Warn("signal engine worker", "err", err)
	}
	select {
	case <-p.done:
	case <-time.After(p.timeout):
		p.log.Warn("engine worker ignored SIGTERM, killing it")
		_ = p.cmd.Process.Kill()
		<-p.done
	}
	p.log.Info("engine worker stopped", "err", p.err)
}
