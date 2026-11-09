// Package config builds the whole gateway configuration from defaults, one
// YAML file, environment variables and command line flags, in that order of
// increasing precedence.
//
// Every setting has an environment variable named WHISPER_HUB_ followed by
// its YAML path in upper case, joined by "_": hub.queue_length is
// WHISPER_HUB_HUB_QUEUE_LENGTH. Lists are comma separated, maps are
// "key=value" pairs separated by commas, and anything else that is not a
// scalar (hub.workers) takes YAML flow syntax:
// WHISPER_HUB_HUB_WORKERS='[{url: "http://w:9100"}]'.
//
// The variables and flags of an existing whisper-asr-webservice deployment
// (and of its sidecar sidecar) are accepted too, see applyCompat, so the
// gateway can replace it without touching the deployment.
package config

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/GRbit/whisper-hub/internal/audio"
	"github.com/GRbit/whisper-hub/internal/auth"
	"github.com/GRbit/whisper-hub/internal/hub"
	"github.com/GRbit/whisper-hub/internal/legacy"
	"github.com/GRbit/whisper-hub/internal/openaiapi"
	"github.com/GRbit/whisper-hub/internal/workerproc"
	"github.com/GRbit/whisper-hub/internal/wyoming"
)

// Config is the gateway configuration: the packages' own Config structs
// plus the few settings only the binary has. Zero values inside the package
// sections mean that package's documented default.
type Config struct {
	HTTP    HTTP             `yaml:"http"`
	Log     Log              `yaml:"log"`
	Audio   audio.Config     `yaml:"audio"`
	Hub     hub.Config       `yaml:"hub"`
	Auth    auth.Config      `yaml:"auth"`
	Legacy  legacy.Config    `yaml:"legacy"`
	OpenAI  openaiapi.Config `yaml:"openai"`
	Wyoming Wyoming          `yaml:"wyoming"`
	// Worker is the engine worker the gateway starts itself (all-in-one
	// image). Default: none.
	Worker workerproc.Config `yaml:"worker"`

	// File is the configuration file that was read, "" for none.
	File string `yaml:"-"`
}

// HTTP configures the HTTP listeners.
type HTTP struct {
	// Listen is the main HTTP address: legacy, OpenAI, /healthz, usage.
	// Default ":9000".
	Listen string `yaml:"listen"`
	// OpenListen, when set, serves the same routes on a second address
	// without authentication, as the webservice port of the old bundled
	// sidecar setup did. Default "": off.
	OpenListen string `yaml:"open_listen"`
	// ShutdownGrace is how long running requests may finish after SIGTERM
	// before they are cut off. Default 30s.
	ShutdownGrace time.Duration `yaml:"shutdown_grace"`
}

// Log configures logging.
type Log struct {
	// Level is debug, info, warn or error. Default info.
	Level string `yaml:"level"`
	// Format is text or json. Default text.
	Format string `yaml:"format"`
}

// Wyoming configures the Wyoming listener.
type Wyoming struct {
	// Enabled starts the Wyoming server. Default true.
	Enabled        bool `yaml:"enabled"`
	wyoming.Config `yaml:",inline"`
}

// DefaultWorker is the engine worker used when none is configured: the
// worker inside the all-in-one image.
const DefaultWorker = "http://127.0.0.1:9100"

// EnvConfigFile names the configuration file when --config is not given.
const EnvConfigFile = "WHISPER_HUB_CONFIG"

const envPrefix = "WHISPER_HUB_"

func defaults() Config {
	return Config{
		HTTP:    HTTP{Listen: ":9000", ShutdownGrace: 30 * time.Second},
		Log:     Log{Level: "info", Format: "text"},
		Hub:     hub.Config{Workers: []hub.WorkerConfig{{URL: DefaultWorker}}},
		Wyoming: Wyoming{Enabled: true},
	}
}

// Load builds the configuration for the serve command from args (flags
// without the program name and subcommand) and the environment getenv
// (os.LookupEnv in production). It returns flag.ErrHelp after printing
// usage to out when asked for help.
func Load(args []string, getenv func(string) (string, bool), out io.Writer) (Config, error) {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(out)
	file := fs.String("config", "", "YAML configuration file (env "+EnvConfigFile+")")
	var host, port string
	// -h is host, not help, because the upstream webservice uses it so.
	fs.StringVar(&host, "host", "", "HTTP listen host (upstream compatible)")
	fs.StringVar(&host, "h", "", "shorthand for --host")
	fs.StringVar(&port, "port", "", "HTTP listen port (upstream compatible)")
	fs.StringVar(&port, "p", "", "shorthand for --port")
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	if fs.NArg() > 0 {
		return Config{}, fmt.Errorf("unexpected arguments %q", fs.Args())
	}

	cfg := defaults()
	if *file == "" {
		*file, _ = getenv(EnvConfigFile)
	}
	if *file != "" {
		if err := readFile(*file, &cfg); err != nil {
			return Config{}, err
		}
		cfg.File = *file
	}
	compatOpen, err := applyCompat(&cfg, getenv)
	if err != nil {
		return Config{}, err
	}
	if err := applyEnv(&cfg, getenv); err != nil {
		return Config{}, err
	}
	// The upstream flags address the webservice port, which is the open
	// listener while the sidecar compatibility mode is on.
	target := &cfg.HTTP.Listen
	if compatOpen {
		target = &cfg.HTTP.OpenListen
	}
	if host != "" || port != "" {
		addr, err := withHostPort(*target, host, port)
		if err != nil {
			return Config{}, err
		}
		*target = addr
	}
	return cfg, cfg.validate()
}

func readFile(path string, cfg *Config) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	// A misspelled key would otherwise be silently ignored.
	dec.KnownFields(true)
	if err := dec.Decode(cfg); err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("config %s: %w", path, err)
	}
	return nil
}

func withHostPort(addr, host, port string) (string, error) {
	h, p, err := net.SplitHostPort(addr)
	if err != nil {
		h, p = "", "9000"
	}
	if host != "" {
		h = host
	}
	if port != "" {
		p = port
	}
	if h == "0.0.0.0" {
		h = "" // both stacks, as the upstream default host means "everywhere"
	}
	return net.JoinHostPort(h, p), nil
}

func (c *Config) validate() error {
	c.Log.Level = strings.ToLower(c.Log.Level)
	switch c.Log.Level {
	case "debug", "info", "warn", "error":
	case "warning":
		c.Log.Level = "warn"
	case "critical", "fatal":
		c.Log.Level = "error"
	default:
		return fmt.Errorf("config: log.level %q is not debug, info, warn or error", c.Log.Level)
	}
	c.Log.Format = strings.ToLower(c.Log.Format)
	if c.Log.Format != "text" && c.Log.Format != "json" {
		return fmt.Errorf("config: log.format %q is not text or json", c.Log.Format)
	}
	if c.HTTP.Listen == "" {
		return errors.New("config: http.listen is empty")
	}
	if c.HTTP.ShutdownGrace <= 0 {
		c.HTTP.ShutdownGrace = 30 * time.Second
	}
	return nil
}

// YAML renders the effective configuration, for the config subcommand.
func (c Config) YAML() ([]byte, error) { return yaml.Marshal(c) }

// applyCompat maps the environment of a whisper-asr-webservice deployment.
// It reports whether the sidecar bundled mode is on, in which case the
// upstream --host/--port flags address the open listener.
//
// SIDECAR_* only count with SIDECAR_ENABLED, as they did upstream.
// SIDECAR_UPSTREAM and SIDECAR_BACKEND_WAIT have no equivalent: there
// is no proxy hop, and the hub tracks worker readiness itself.
func applyCompat(c *Config, getenv func(string) (string, bool)) (bool, error) {
	str := func(name string, dst *string) {
		if v, ok := getenv(name); ok && v != "" {
			*dst = v
		}
	}
	str("UPLOAD_SPOOL_DIR", &c.Audio.SpoolDir)
	str("DEBUG_FAILED_UPLOADS_DIR", &c.Audio.FailedUploadsDir)
	_, hasLevel := getenv("LOG_LEVEL")
	str("LOG_LEVEL", &c.Log.Level)
	if v, ok := getenv("SAMPLE_RATE"); ok && v != "" {
		if err := setValue(&c.Legacy.RawSampleRate, v); err != nil {
			return false, fmt.Errorf("config: SAMPLE_RATE: %w", err)
		}
	}

	enabled, _ := getenv("SIDECAR_ENABLED")
	switch strings.ToLower(enabled) {
	case "true", "1", "yes":
	default:
		return false, nil
	}
	c.Auth.UsersFile = "/config/auth.yaml"
	str("SIDECAR_CONFIG", &c.Auth.UsersFile)
	c.HTTP.Listen = ":9001"
	str("SIDECAR_LISTEN", &c.HTTP.Listen)
	host, port := "", "9000"
	str("WHISPER_HOST", &host)
	str("WHISPER_PORT", &port)
	open, err := withHostPort(":9000", host, port)
	if err != nil {
		return false, err
	}
	c.HTTP.OpenListen = open
	dir := "/data/sidecar"
	str("SIDECAR_USAGE_DIR", &dir)
	c.Auth.Usage.TotalsFile = filepath.Join(dir, "usage.json")
	c.Auth.Usage.LogFile = filepath.Join(dir, "usage.jsonl")
	str("SIDECAR_REALM", &c.Auth.Realm)
	if !hasLevel {
		str("SIDECAR_LOG_LEVEL", &c.Log.Level)
	}
	for name, dst := range map[string]any{
		"SIDECAR_CONFIG_POLL":       &c.Auth.PollInterval,
		"SIDECAR_SNAPSHOT_INTERVAL": &c.Auth.Usage.SnapshotInterval,
		"SIDECAR_MAX_UPLOAD_BYTES":  &c.Legacy.MaxUploadBytes,
		"SIDECAR_TRUST_FORWARDED":   &c.Auth.TrustForwarded,
	} {
		if v, ok := getenv(name); ok && v != "" {
			if err := setValue(dst, v); err != nil {
				return false, fmt.Errorf("config: %s: %w", name, err)
			}
		}
	}
	// The proxy's limit covered every upload it saw; OpenAI is new here,
	// but the same limit is the least surprising choice.
	c.OpenAI.MaxUploadBytes = c.Legacy.MaxUploadBytes
	return true, nil
}
