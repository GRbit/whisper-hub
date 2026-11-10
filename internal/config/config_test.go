package config_test

import (
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/GRbit/whisper-hub/internal/config"
)

func envOf(kv ...string) func(string) (string, bool) {
	m := map[string]string{}
	for i := 0; i < len(kv); i += 2 {
		m[kv[i]] = kv[i+1]
	}
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

func load(t *testing.T, args []string, env func(string) (string, bool)) config.Config {
	t.Helper()
	cfg, err := config.Load(args, env, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func writeFile(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestZeroConfigDefaults(t *testing.T) {
	cfg := load(t, nil, envOf())
	if cfg.HTTP.Listen != ":9000" || cfg.HTTP.OpenListen != "" || !cfg.Wyoming.Enabled || cfg.Auth.UsersFile != "" {
		t.Fatalf("%+v", cfg)
	}
	if len(cfg.Hub.Workers) != 1 || cfg.Hub.Workers[0].URL != config.DefaultWorker {
		t.Fatalf("workers %+v", cfg.Hub.Workers)
	}
	if cfg.Log.Level != "info" || cfg.Log.Format != "text" || cfg.File != "" {
		t.Fatalf("log %+v", cfg.Log)
	}
}

// Precedence: defaults < file < compat env < WHISPER_HUB_* env < flags.
func TestPrecedence(t *testing.T) {
	file := writeFile(t, `
http: {listen: ":1000"}
log: {level: debug}
audio: {spool_dir: /from-file}
hub:
  workers: [{url: "http://a:9100", max_in_flight: 2}, {url: "http://b:9100"}]
  aliases: {big: large-v3}
wyoming: {listen: ":2000", stream_transcripts: true}
`)
	cfg := load(t, []string{"--config", file}, envOf())
	if cfg.HTTP.Listen != ":1000" || cfg.Log.Level != "debug" || cfg.Wyoming.Listen != ":2000" || !cfg.Wyoming.StreamTranscripts {
		t.Fatalf("file not applied: %+v", cfg)
	}
	if len(cfg.Hub.Workers) != 2 || cfg.Hub.Workers[0].MaxInFlight != 2 || cfg.Hub.Aliases["big"] != "large-v3" || cfg.File != file {
		t.Fatalf("hub %+v", cfg.Hub)
	}

	cfg = load(t, []string{"-p", "3000"}, envOf(config.EnvConfigFile, file,
		"UPLOAD_SPOOL_DIR", "/compat",
		"LOG_LEVEL", "WARNING",
		"WHISPER_HUB_HTTP_LISTEN", "127.0.0.1:4000",
		"WHISPER_HUB_AUDIO_SPOOL_DIR", "/native",
	))
	if cfg.HTTP.Listen != "127.0.0.1:3000" {
		t.Errorf("flag over env: listen %q", cfg.HTTP.Listen)
	}
	if cfg.Audio.SpoolDir != "/native" || cfg.Log.Level != "warn" {
		t.Errorf("env order: spool %q level %q", cfg.Audio.SpoolDir, cfg.Log.Level)
	}
}

func TestEnvValueTypes(t *testing.T) {
	cfg := load(t, nil, envOf(
		"WHISPER_HUB_HUB_WORKERS", `[{url: "http://w1:9100"}, {url: "http://w2:9100", max_in_flight: 3}]`,
		"WHISPER_HUB_HUB_ALIASES", "whisper-1=small, big=large-v3",
		"WHISPER_HUB_HUB_QUEUE_LENGTH", "7",
		"WHISPER_HUB_HUB_PRIORITY_AGING", "90s",
		"WHISPER_HUB_AUTH_PROTECTED_PATHS", "/asr, /v1/",
		"WHISPER_HUB_AUTH_TRUST_FORWARDED", "true",
		"WHISPER_HUB_AUTH_WYOMING_ALLOW_CIDRS", "[10.0.0.0/8]",
		"WHISPER_HUB_AUTH_USAGE_LOG_MAX_BYTES", "1048576",
		"WHISPER_HUB_WYOMING_ENABLED", "false",
		"WHISPER_HUB_WYOMING_ZEROCONF_NAME", "kitchen",
	))
	h := cfg.Hub
	if len(h.Workers) != 2 || h.Workers[1].MaxInFlight != 3 || h.Aliases["whisper-1"] != "small" || h.Aliases["big"] != "large-v3" ||
		h.QueueLength != 7 || h.PriorityAging != 90*time.Second {
		t.Fatalf("hub %+v", h)
	}
	a := cfg.Auth
	if !slices.Equal(a.ProtectedPaths, []string{"/asr", "/v1/"}) || !a.TrustForwarded ||
		!slices.Equal(a.Wyoming.AllowCIDRs, []string{"10.0.0.0/8"}) || a.Usage.LogMaxBytes != 1<<20 {
		t.Fatalf("auth %+v", a)
	}
	if cfg.Wyoming.Enabled || cfg.Wyoming.Zeroconf.Name != "kitchen" {
		t.Fatalf("wyoming %+v", cfg.Wyoming)
	}
}

func TestErrorsAreReported(t *testing.T) {
	for name, tc := range map[string]struct {
		args []string
		env  func(string) (string, bool)
	}{
		"bad duration":  {nil, envOf("WHISPER_HUB_HUB_POLL_INTERVAL", "soon")},
		"bad int":       {nil, envOf("WHISPER_HUB_HUB_QUEUE_LENGTH", "many")},
		"bad worker":    {nil, envOf("WHISPER_HUB_HUB_WORKERS", `[{adress: x}]`)},
		"bad map":       {nil, envOf("WHISPER_HUB_HUB_ALIASES", "whisper-1")},
		"bad level":     {nil, envOf("LOG_LEVEL", "loud")},
		"bad format":    {nil, envOf("WHISPER_HUB_LOG_FORMAT", "xml")},
		"missing file":  {[]string{"--config", "/nonexistent.yaml"}, envOf()},
		"unknown key":   {[]string{"--config", writeFile(t, "hub: {queue_lenght: 3}\n")}, envOf()},
		"stray args":    {[]string{"extra"}, envOf()},
		"unknown flag":  {[]string{"--model", "x"}, envOf()},
		"bad proxy var": {nil, envOf("SIDECAR_ENABLED", "true", "SIDECAR_CONFIG_POLL", "5")},
	} {
		if _, err := config.Load(tc.args, tc.env, io.Discard); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := config.Load([]string{"--help"}, envOf(), io.Discard); !errors.Is(err, flag.ErrHelp) {
		t.Errorf("--help: %v", err)
	}
}

// The upstream image is started as "whisper-asr-webservice --host 0.0.0.0
// --port 9000"; those flags must keep working.
func TestUpstreamFlagsAndEnv(t *testing.T) {
	cfg := load(t, []string{"--host", "0.0.0.0", "--port", "9100"}, envOf(
		"LOG_LEVEL", "DEBUG", "UPLOAD_SPOOL_DIR", "/var/tmp/up", "DEBUG_FAILED_UPLOADS_DIR", "/var/tmp/failed", "SAMPLE_RATE", "8000",
		"ASR_MODEL", "base", "ASR_ENGINE", "faster_whisper", // worker settings: ignored here
	))
	if cfg.HTTP.Listen != ":9100" || cfg.Log.Level != "debug" || cfg.Audio.SpoolDir != "/var/tmp/up" ||
		cfg.Audio.FailedUploadsDir != "/var/tmp/failed" || cfg.Legacy.RawSampleRate != 8000 {
		t.Fatalf("%+v", cfg)
	}
	cfg = load(t, []string{"-h", "127.0.0.1"}, envOf())
	if cfg.HTTP.Listen != "127.0.0.1:9000" {
		t.Fatalf("listen %q", cfg.HTTP.Listen)
	}
}

func TestSidecarCompat(t *testing.T) {
	// Without SIDECAR_ENABLED the proxy variables have no effect, as upstream.
	cfg := load(t, nil, envOf("SIDECAR_CONFIG", "/x.yaml", "SIDECAR_LISTEN", ":1"))
	if cfg.Auth.UsersFile != "" || cfg.HTTP.Listen != ":9000" {
		t.Fatalf("proxy vars applied while disabled: %+v", cfg)
	}

	cfg = load(t, nil, envOf("SIDECAR_ENABLED", "true"))
	if cfg.Auth.UsersFile != "/config/auth.yaml" || cfg.HTTP.Listen != ":9001" || cfg.HTTP.OpenListen != ":9000" ||
		cfg.Auth.Usage.TotalsFile != "/data/sidecar/usage.json" || cfg.Auth.Usage.LogFile != "/data/sidecar/usage.jsonl" {
		t.Fatalf("bundled defaults: %+v", cfg)
	}

	cfg = load(t, []string{"--port", "9200"}, envOf("SIDECAR_ENABLED", "1",
		"SIDECAR_CONFIG", "/etc/auth.yaml", "SIDECAR_LISTEN", "0.0.0.0:8443",
		"SIDECAR_USAGE_DIR", "/data/u", "SIDECAR_CONFIG_POLL", "10s", "SIDECAR_SNAPSHOT_INTERVAL", "1m",
		"SIDECAR_REALM", "office", "SIDECAR_MAX_UPLOAD_BYTES", "1000", "SIDECAR_TRUST_FORWARDED", "true",
		"SIDECAR_LOG_LEVEL", "debug", "WHISPER_HOST", "127.0.0.1",
	))
	a := cfg.Auth
	if a.UsersFile != "/etc/auth.yaml" || cfg.HTTP.Listen != "0.0.0.0:8443" || a.PollInterval != 10*time.Second ||
		a.Usage.SnapshotInterval != time.Minute || a.Realm != "office" || cfg.Legacy.MaxUploadBytes != 1000 || cfg.OpenAI.MaxUploadBytes != 1000 ||
		!a.TrustForwarded || cfg.Log.Level != "debug" || a.Usage.TotalsFile != "/data/u/usage.json" {
		t.Fatalf("%+v", cfg)
	}
	// --port addresses the open webservice port in bundled mode.
	if cfg.HTTP.OpenListen != "127.0.0.1:9200" {
		t.Fatalf("open listen %q", cfg.HTTP.OpenListen)
	}
	// LOG_LEVEL wins over the proxy's own level.
	cfg = load(t, nil, envOf("SIDECAR_ENABLED", "true", "SIDECAR_LOG_LEVEL", "debug", "LOG_LEVEL", "ERROR"))
	if cfg.Log.Level != "error" {
		t.Fatalf("level %q", cfg.Log.Level)
	}
}

func TestExampleConfigParses(t *testing.T) {
	_, self, _, _ := runtime.Caller(0)
	example := filepath.Join(filepath.Dir(self), "..", "..", "deploy", "whisper-hub.example.yaml")
	cfg := load(t, []string{"--config", example}, envOf())
	if len(cfg.Hub.Workers) == 0 || cfg.HTTP.Listen == "" {
		t.Fatalf("%+v", cfg)
	}
	out, err := cfg.YAML()
	if err != nil || !strings.Contains(string(out), "queue_length:") {
		t.Fatalf("yaml: %v %s", err, out)
	}
	// The rendered configuration is itself a valid configuration file.
	again := load(t, []string{"--config", writeFile(t, string(out))}, envOf())
	if again.Hub.QueueLength != cfg.Hub.QueueLength || again.HTTP.Listen != cfg.HTTP.Listen {
		t.Fatalf("round trip: %+v", again)
	}
}

// zeroByDefault lists the settings whose default really is zero, empty or
// false. Any other setting that renders as zero for an empty configuration
// is a default its package resolves but YAML does not show.
var zeroByDefault = map[string]bool{
	"http.open_listen":           true,
	"audio.failed_uploads_dir":   true,
	"hub.default_model":          true, // chosen at run time from the workers
	"hub.aliases.whisper-1":      true, // "" is the default model
	"auth.users_file":            true,
	"auth.trust_forwarded":       true,
	"auth.anonymous.restrict":    true,
	"auth.anonymous.models":      true,
	"auth.wyoming.user":          true,
	"auth.wyoming.allow_cidrs":   true,
	"auth.usage.totals_file":     true,
	"auth.usage.log_file":        true,
	"auth.usage.log_max_bytes":   true,
	"legacy.max_upload_bytes":    true,
	"openai.max_upload_bytes":    true,
	"wyoming.version":            true, // filled in by the binary with its own version
	"wyoming.stream_transcripts": true,
	"wyoming.zeroconf.enabled":   true,
	"wyoming.zeroconf.host":      true,
	"worker.command":             true,
	"worker.engine":              true,
}

// leaves flattens a decoded YAML document into "a.b.c" -> value.
func leaves(prefix string, v any, out map[string]any) {
	m, ok := v.(map[string]any)
	if !ok {
		out[prefix] = v
		return
	}
	for k, sub := range m {
		p := k
		if prefix != "" {
			p = prefix + "." + k
		}
		leaves(p, sub, out)
	}
}

func renderedLeaves(t *testing.T, cfg config.Config) map[string]any {
	t.Helper()
	out, err := cfg.YAML()
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := yaml.Unmarshal(out, &doc); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	flat := map[string]any{}
	leaves("", doc, flat)
	return flat
}

func isZero(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == "" || x == "0s"
	case int:
		return x == 0
	case float64:
		return x == 0
	case bool:
		return !x
	case []any:
		return len(x) == 0
	case map[string]any:
		return len(x) == 0
	}
	return false
}

// The config subcommand shows what the service runs with, so a default
// resolved inside a package must not print as its zero value.
func TestYAMLShowsResolvedDefaults(t *testing.T) {
	flat := renderedLeaves(t, load(t, nil, envOf()))
	var keys []string
	for k := range flat {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		if isZero(flat[k]) && !zeroByDefault[k] {
			t.Errorf("%s renders as %v for an empty configuration", k, flat[k])
		}
	}
	workers, _ := flat["hub.workers"].([]any)
	if len(workers) != 1 || isZero(workers[0].(map[string]any)["max_in_flight"]) {
		t.Errorf("hub.workers = %v, want max_in_flight resolved", flat["hub.workers"])
	}
	for k := range zeroByDefault {
		if _, ok := flat[k]; !ok {
			t.Errorf("zeroByDefault names %s, which is not a setting", k)
		}
	}
}

// Rendering resolves zero values only: what was set stays as set, and the
// printed keys are exactly those of the example file.
func TestYAMLKeepsSetValuesAndKeys(t *testing.T) {
	flat := renderedLeaves(t, load(t, nil, envOf(
		"WHISPER_HUB_HUB_QUEUE_LENGTH", "7",
		"WHISPER_HUB_HUB_PRIORITY_AGING", "-1s",
		"WHISPER_HUB_AUTH_RATE_LIMIT_FAILURES", "-1",
		"WHISPER_HUB_WYOMING_MAX_CONNECTIONS", "-1",
		"WHISPER_HUB_WYOMING_PROGRAM_NAME", "kitchen",
	)))
	for k, want := range map[string]any{
		"hub.queue_length": 7, "hub.priority_aging": "-1s", "auth.rate_limit.failures": -1,
		"wyoming.max_connections": -1, "wyoming.program_name": "kitchen", "wyoming.zeroconf.name": "kitchen",
	} {
		if flat[k] != want {
			t.Errorf("%s = %v, want %v", k, flat[k], want)
		}
	}

	_, self, _, _ := runtime.Caller(0)
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(self), "..", "..", "deploy", "whisper-hub.example.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	example := map[string]any{}
	leaves("", doc, example)
	for k := range flat {
		if _, ok := example[k]; !ok {
			t.Errorf("%s is printed but missing from the example file", k)
		}
	}
	for k := range example {
		if _, ok := flat[k]; !ok {
			t.Errorf("%s is in the example file but not printed", k)
		}
	}
}
