package wyoming

import (
	"fmt"
	"time"
)

// Config configures the Wyoming server. Zero values select the documented
// defaults; for the durations a negative value disables the limit.
type Config struct {
	// Listen is the TCP address for ListenAndServe. Default ":10300".
	Listen string `yaml:"listen"`
	// ProgramName is the asr program name in info; Home Assistant shows it
	// as the entity name. Default "whisper-hub".
	ProgramName string `yaml:"program_name"`
	// Version is reported as the program version in info. Default empty
	// (sent as null).
	Version string `yaml:"version"`
	// StreamTranscripts advertises supports_transcript_streaming and sends
	// transcript-start / transcript-chunk / transcript-stop around the final
	// transcript. Default false.
	//
	// Off by default: Home Assistant's STT client reads only transcript and
	// error, and the chunks can only start after audio-stop because the
	// engine gets the whole utterance, so the extra events buy HA nothing.
	StreamTranscripts bool `yaml:"stream_transcripts"`

	// MaxHeaderBytes caps one JSON header line. Default 256 KiB.
	MaxHeaderBytes int `yaml:"max_header_bytes"`
	// MaxDataBytes caps one additional data block. Default 1 MiB.
	MaxDataBytes int `yaml:"max_data_bytes"`
	// MaxPayloadBytes caps one binary payload (one audio chunk). Default 4 MiB.
	MaxPayloadBytes int `yaml:"max_payload_bytes"`
	// MaxUtteranceBytes caps raw audio buffered for one utterance, before
	// conversion. Default 64 MiB.
	MaxUtteranceBytes int64 `yaml:"max_utterance_bytes"`
	// MaxUtteranceDuration caps the audio length of one utterance. Default 5m.
	MaxUtteranceDuration time.Duration `yaml:"max_utterance_duration"`
	// IdleTimeout closes a connection that sends no event for this long
	// while no transcription is pending. Default 5m.
	IdleTimeout time.Duration `yaml:"idle_timeout"`
	// WriteTimeout bounds each event write to a client. Default 30s.
	WriteTimeout time.Duration `yaml:"write_timeout"`
	// MaxConnections caps concurrent connections; extra ones get an error
	// event and are closed. Default 64; negative means unlimited.
	MaxConnections int `yaml:"max_connections"`
	// ShutdownGrace is how long in-flight transcriptions may finish after
	// the Serve context is canceled. Default 10s.
	ShutdownGrace time.Duration `yaml:"shutdown_grace"`

	// Zeroconf configures the mDNS announcement for Home Assistant
	// discovery.
	Zeroconf ZeroconfConfig `yaml:"zeroconf"`
}

// ZeroconfConfig configures the _wyoming._tcp.local. mDNS announcement.
//
// It only works when the server shares a layer 2 network with Home
// Assistant (host networking); behind docker bridge NAT nobody hears it.
type ZeroconfConfig struct {
	// Enabled turns the announcement on. Default false.
	Enabled bool `yaml:"enabled"`
	// Name is the service instance name. Default "whisper-hub".
	Name string `yaml:"name"`
	// Host is the IPv4 address to advertise. Default: the address of the
	// interface that routes to the mDNS group.
	Host string `yaml:"host"`
}

// Defaults, as documented on the Config fields.
const (
	defaultListen               = ":10300"
	defaultProgramName          = "whisper-hub"
	defaultMaxHeaderBytes       = 256 << 10
	defaultMaxDataBytes         = 1 << 20
	defaultMaxPayloadBytes      = 4 << 20
	defaultMaxUtteranceBytes    = 64 << 20
	defaultMaxUtteranceDuration = 5 * time.Minute
	defaultIdleTimeout          = 5 * time.Minute
	defaultWriteTimeout         = 30 * time.Second
	defaultShutdownGrace        = 10 * time.Second
	defaultMaxConnections       = 64
)

// Resolved returns c with every zero value replaced by the default it
// stands for, as New uses it. It does not validate; New does.
func (c Config) Resolved() Config {
	if c.Listen == "" {
		c.Listen = defaultListen
	}
	if c.ProgramName == "" {
		c.ProgramName = defaultProgramName
	}
	if c.Zeroconf.Name == "" {
		c.Zeroconf.Name = c.ProgramName
	}
	intDefault(&c.MaxHeaderBytes, defaultMaxHeaderBytes)
	intDefault(&c.MaxDataBytes, defaultMaxDataBytes)
	intDefault(&c.MaxPayloadBytes, defaultMaxPayloadBytes)
	if c.MaxUtteranceBytes == 0 {
		c.MaxUtteranceBytes = defaultMaxUtteranceBytes
	}
	intDefault(&c.MaxConnections, defaultMaxConnections)
	durDefault(&c.MaxUtteranceDuration, defaultMaxUtteranceDuration)
	durDefault(&c.IdleTimeout, defaultIdleTimeout)
	durDefault(&c.WriteTimeout, defaultWriteTimeout)
	durDefault(&c.ShutdownGrace, defaultShutdownGrace)
	return c
}

func (c Config) withDefaults() (Config, error) {
	c = c.Resolved()
	if c.MaxHeaderBytes < 0 || c.MaxDataBytes < 0 || c.MaxPayloadBytes < 0 || c.MaxUtteranceBytes < 0 {
		return c, fmt.Errorf("wyoming: byte limits must not be negative")
	}
	return c, nil
}

func intDefault(v *int, def int) {
	if *v == 0 {
		*v = def
	}
}

func durDefault(v *time.Duration, def time.Duration) {
	if *v == 0 {
		*v = def
	}
}

// enabled reports whether a duration limit is on (negative disables it).
func enabled(d time.Duration) bool { return d > 0 }
