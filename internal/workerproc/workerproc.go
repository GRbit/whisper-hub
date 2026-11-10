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
	// Engine is the engine installed next to Command. When set, the engine
	// the worker is asked for (EngineVar or EngineAlias) must be unset (it
	// is then set to Engine) or equal to it. "all" means every real engine
	// is installed: the variable picks one and defaults to openai_whisper,
	// as upstream does. Default "": no check.
	Engine string `yaml:"engine"`
	// ImageTag is how the error for a missing engine names the image to
	// use; "%s" is replaced by the requested engine. Default
	// "whisper-hub:%s".
	ImageTag string `yaml:"image_tag"`
	// StopTimeout is how long the worker may take to exit after SIGTERM
	// before it is killed. Default 10s.
	StopTimeout time.Duration `yaml:"stop_timeout"`
}

// Engines are the engine names the worker knows.
var Engines = []string{"faster_whisper", "whisperx", "openai_whisper", "fake"}

// realEngines are the engines an "all" image contains.
var realEngines = Engines[:3]

// EngineAll is Config.Engine for an image with every real engine.
const EngineAll = "all"

// UpstreamDefaultEngine is what whisper-asr-webservice runs when no engine
// is selected.
const UpstreamDefaultEngine = "openai_whisper"

// Proc is a running worker process.
type Proc struct {
	cmd     *exec.Cmd
	log     *slog.Logger
	timeout time.Duration
	done    chan struct{}
	err     error
}

// Resolved returns c with every zero value replaced by the default it
// stands for, as Env and Start use it. An empty Command stays empty.
func (c Config) Resolved() Config {
	if c.ImageTag == "" {
		c.ImageTag = "whisper-hub:%s"
	}
	if c.StopTimeout <= 0 {
		c.StopTimeout = 10 * time.Second
	}
	return c
}

// EngineVar selects the worker's engine. EngineAlias is the
// whisper-asr-webservice name, accepted so that deployment keeps working;
// EngineVar wins when both are set, and an empty value counts as unset, as
// in the worker.
const (
	EngineVar   = "WHISPER_HUB_ENGINE"
	EngineAlias = "ASR_ENGINE"
)

// requestedEngine returns the variable that selects the engine and its value.
func requestedEngine(env []string) (name, value string) {
	vals := map[string]string{}
	for _, kv := range env {
		for _, n := range []string{EngineVar, EngineAlias} {
			if v, ok := strings.CutPrefix(kv, n+"="); ok {
				vals[n] = strings.TrimSpace(v) // last one wins, as in exec
			}
		}
	}
	for _, n := range []string{EngineVar, EngineAlias} {
		if vals[n] != "" {
			return n, vals[n]
		}
	}
	return EngineVar, ""
}

// Env returns env adjusted for the worker: the engine defaults to the
// installed one instead of upstream's openai_whisper, and asking for an
// engine the image lacks is an error naming the image that has it.
func (c Config) Env(env []string) ([]string, error) {
	name, requested := requestedEngine(env)
	if c.Engine == "" {
		return env, nil
	}
	installed, def := []string{c.Engine}, c.Engine
	if c.Engine == EngineAll {
		installed, def = realEngines, UpstreamDefaultEngine
	}
	if requested == "" {
		return append(slices.Clone(env), EngineVar+"="+def), nil
	}
	if slices.Contains(installed, requested) {
		return env, nil
	}
	tag := c.Resolved().ImageTag
	if !slices.Contains(Engines, requested) {
		return nil, fmt.Errorf("%s=%q is not an engine (%s)", name, requested, strings.Join(Engines, ", "))
	}
	return nil, fmt.Errorf("%s=%s, but this image contains the %s engine only: use the %s or %s image, or unset %s",
		name, requested, c.Engine, strings.ReplaceAll(tag, "%s", requested), strings.ReplaceAll(tag, "%s", EngineAll), name)
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
	p := &Proc{cmd: cmd, log: log, timeout: c.Resolved().StopTimeout, done: make(chan struct{})}
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
