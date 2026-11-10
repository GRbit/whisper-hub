package wyoming

import (
	"bytes"
	"context"
	"errors"
	"net"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
)

func TestDescribeInfo(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{Version: "1.2.3"}, svc, nil)
	c := h.dial()
	c.send(typeDescribe, nil, nil)
	ev := c.recv()
	if ev.Type != typeInfo {
		t.Fatalf("got %s", ev.Type)
	}
	// Every service array is present, as Info.event() sends them.
	for _, k := range []string{"asr", "tts", "handle", "intent", "wake", "mic", "snd"} {
		if string(ev.Data[k]) == "" || string(ev.Data[k]) == "null" {
			t.Errorf("info lacks %q array", k)
		}
	}
	info := decode[infoData](t, ev)
	if len(info.ASR) != 1 {
		t.Fatalf("want 1 asr program, got %d", len(info.ASR))
	}
	p := info.ASR[0]
	if p.Name != defaultProgramName || !p.Installed || p.SupportsTranscriptStreaming || !p.RequiresExternalVAD ||
		p.Version == nil || *p.Version != "1.2.3" {
		t.Errorf("program: %+v", p)
	}
	if len(p.Models) != 2 || p.Models[0].Name != "tiny" || p.Models[1].Name != "small" {
		t.Fatalf("models: %+v", p.Models)
	}
	m := p.Models[0]
	if !m.Installed || m.Attribution.Name != "SYSTRAN" || m.Description == nil {
		t.Errorf("model: %+v", m)
	}
	if !slices.Contains(m.Languages, "zh-HK") || !slices.Contains(m.Languages, "yue") {
		t.Errorf("Cantonese tag not advertised: %v", m.Languages)
	}
	if slices.Contains(p.Models[1].Languages, "zh-HK") {
		t.Errorf("zh-HK advertised for a model without yue")
	}

	// Raw check for keys Info.from_event needs even when null.
	for _, k := range []string{`"installed":true`, `"description":"tiny (faster_whisper)","version":"1.2.3"`, `"supports_transcript_streaming":false`,
		`"prefers_auto_gain_enabled":true`, `"prefers_noise_reduction_enabled":true`} {
		if !strings.Contains(string(ev.Data["asr"]), k) {
			t.Errorf("asr JSON lacks %s: %s", k, ev.Data["asr"])
		}
	}
}

func TestDescribeNoModels(t *testing.T) {
	svc := newFakeService()
	svc.models = nil
	h := startServer(t, Config{StreamTranscripts: true}, svc, nil)
	c := h.dial()
	c.send(typeDescribe, nil, nil)
	info := decode[infoData](t, c.recv())
	p := info.ASR[0]
	if p.Installed || p.Models == nil || len(p.Models) != 0 || !p.SupportsTranscriptStreaming {
		t.Errorf("program: %+v", p)
	}
}

func TestTranscribeRoundTrip(t *testing.T) {
	svc := newFakeService()
	id := &core.Identity{User: "kitchen"}
	h := startServer(t, Config{}, svc, func(net.Addr) (*core.Identity, error) { return id, nil })
	c := h.dial()
	c.send(typeTranscribe, map[string]any{"name": "small", "language": "en-US"}, nil)
	a, b := pcm(3200), pcm(1601) // odd tail: one partial frame to drop
	c.utterance(16000, 2, 1, a, b)
	ev := c.recv()
	if ev.Type != typeTranscript {
		t.Fatalf("got %s %v", ev.Type, ev.Data)
	}
	tr := decode[transcriptData](t, ev)
	if tr.Text != "Hello world." || tr.Language != "en" {
		t.Errorf("transcript %+v", tr)
	}
	calls := svc.Calls()
	if len(calls) != 1 {
		t.Fatalf("calls: %d", len(calls))
	}
	req := calls[0].req
	if req.Model != "small" || req.Language != "en" || req.Priority != core.PriorityInteractive ||
		req.Protocol != core.ProtocolWyoming || req.Task != core.TaskTranscribe {
		t.Errorf("request %+v", req)
	}
	want := append(append([]byte(nil), a...), b[:1600]...)
	if !bytes.Equal(req.Audio, want) {
		t.Errorf("audio: got %d bytes, want %d", len(req.Audio), len(want))
	}
	if calls[0].id != id {
		t.Errorf("identity not in Transcribe ctx: %+v", calls[0].id)
	}
}

func TestStreamingEvents(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{StreamTranscripts: true}, svc, nil)
	c := h.dial()
	c.send(typeTranscribe, map[string]any{"language": "nl"}, nil)
	c.utterance(16000, 2, 1, pcm(320))

	var got []string
	for {
		ev := c.recv()
		switch ev.Type {
		case typeTranscriptChunk:
			got = append(got, "chunk:"+decode[transcriptChunkData](t, ev).Text)
		case typeTranscript:
			tr := decode[transcriptData](t, ev)
			got = append(got, "transcript:"+tr.Text+":"+tr.Language)
		case typeTranscriptStart:
			got = append(got, "start:"+decode[transcriptStartData](t, ev).Language)
		default:
			got = append(got, ev.Type)
		}
		if ev.Type == typeTranscriptStop {
			break
		}
	}
	// Chunks are snapshots of the transcript so far (wyoming >= 1.11), and
	// transcript comes before transcript-stop as the README's flow requires.
	want := []string{"start:nl", "chunk:Hello", "chunk:Hello world.", "transcript:Hello world.:nl", typeTranscriptStop}
	if !slices.Equal(got, want) {
		t.Errorf("events\n got %v\nwant %v", got, want)
	}
}

func TestStreamingNoSegmentsStillWrapsTranscript(t *testing.T) {
	svc := newFakeService()
	svc.fn = func(_ context.Context, req *core.Request) (*core.Result, error) {
		return &core.Result{Text: " Done", Language: "de"}, nil
	}
	h := startServer(t, Config{StreamTranscripts: true}, svc, nil)
	c := h.dial()
	c.utterance(16000, 2, 1, pcm(320))
	var types []string
	for len(types) == 0 || types[len(types)-1] != typeTranscriptStop {
		types = append(types, c.recv().Type)
	}
	want := []string{typeTranscriptStart, typeTranscriptChunk, typeTranscript, typeTranscriptStop}
	if !slices.Equal(types, want) {
		t.Errorf("got %v want %v", types, want)
	}
}

func TestSeveralUtterancesAndDefaults(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	c.send(typeTranscribe, map[string]any{"name": "small", "language": "de"}, nil)
	c.utterance(16000, 2, 1, pcm(320))
	if ev := c.recv(); ev.Type != typeTranscript {
		t.Fatalf("first: %s", ev.Type)
	}
	// No transcribe this time: settings do not leak from the previous one.
	c.utterance(16000, 2, 1, pcm(640))
	if ev := c.recv(); ev.Type != typeTranscript {
		t.Fatalf("second: %s", ev.Type)
	}
	// Describe at any time, including between utterances.
	c.send(typeDescribe, nil, nil)
	if ev := c.recv(); ev.Type != typeInfo {
		t.Fatalf("describe: %s", ev.Type)
	}
	calls := svc.Calls()
	if len(calls) != 2 {
		t.Fatalf("calls %d", len(calls))
	}
	if calls[0].req.Model != "small" || calls[0].req.Language != "de" {
		t.Errorf("first req %+v", calls[0].req)
	}
	if calls[1].req.Model != "" || calls[1].req.Language != "" || len(calls[1].req.Audio) != 640 {
		t.Errorf("second req %+v", calls[1].req)
	}
}

func TestChunksWithoutAudioStartAndLateTranscribe(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	f := map[string]any{"rate": 16000, "width": 2, "channels": 1}
	c.send(typeAudioChunk, f, pcm(320))
	c.send(typeTranscribe, map[string]any{"language": "pt-BR"}, nil)
	c.send(typeAudioChunk, f, pcm(320))
	c.send(typeAudioStop, nil, nil)
	if ev := c.recv(); ev.Type != typeTranscript {
		t.Fatalf("got %s", ev.Type)
	}
	req := svc.Calls()[0].req
	if req.Language != "pt" || len(req.Audio) != 640 {
		t.Errorf("req lang %q audio %d", req.Language, len(req.Audio))
	}
}

func TestAudioFormats(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	// 8-bit stereo 48 kHz, then a format switch mid-stream.
	c.send(typeAudioStart, map[string]any{"rate": 48000, "width": 1, "channels": 2}, nil)
	c.send(typeAudioChunk, map[string]any{"rate": 48000, "width": 1, "channels": 2}, []byte{0x80, 0x81, 0x7f, 0x00})
	c.send(typeAudioChunk, map[string]any{"rate": 8000, "width": 2, "channels": 1}, pcm(16))
	c.send(typeAudioStop, nil, nil)
	if ev := c.recv(); ev.Type != typeTranscript {
		t.Fatalf("got %s", ev.Type)
	}
	calls := h.dec.Calls()
	if len(calls) != 2 {
		t.Fatalf("FromRaw calls: %+v", calls)
	}
	if calls[0].rate != 48000 || calls[0].width != 1 || calls[0].channels != 2 ||
		!bytes.Equal(calls[0].data, []byte{0x00, 0x01, 0xff, 0x80}) {
		t.Errorf("8-bit call not converted to signed: %+v", calls[0])
	}
	if calls[1].rate != 8000 || calls[1].width != 2 || len(calls[1].data) != 16 {
		t.Errorf("second call %+v", calls[1])
	}
	// 2 frames at 48k -> 0 samples (integer math in fake), 8 frames at 8k -> 16 samples.
	if got := len(svc.Calls()[0].req.Audio); got != 32 {
		t.Errorf("combined PCM %d bytes", got)
	}
}

func TestBadAudioFormatRejected(t *testing.T) {
	h := startServer(t, Config{}, newFakeService(), nil)
	c := h.dial()
	c.send(typeAudioStart, map[string]any{"rate": 16000, "width": 3, "channels": 0}, nil)
	ev := c.recv()
	if ev.Type != typeError || decode[errorData](t, ev).Code != "bad_audio" {
		t.Fatalf("got %s %v", ev.Type, ev.Data)
	}
	c.expectClosed()
}

func TestEmptyUtterance(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	c.send(typeAudioStart, map[string]any{"rate": 16000, "width": 2, "channels": 1}, nil)
	c.send(typeAudioStop, nil, nil)
	c.send(typeAudioStop, nil, nil) // no stream at all
	for range 2 {
		ev := c.recv()
		if ev.Type != typeTranscript || decode[transcriptData](t, ev).Text != "" {
			t.Fatalf("got %s %v", ev.Type, ev.Data)
		}
	}
	if n := len(svc.Calls()); n != 0 {
		t.Errorf("service called %d times for empty audio", n)
	}
}

func TestDescribeDuringTranscription(t *testing.T) {
	svc := newFakeService()
	release := make(chan struct{})
	svc.fn = func(ctx context.Context, req *core.Request) (*core.Result, error) {
		<-release
		return &core.Result{Text: "ok"}, nil
	}
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	c.utterance(16000, 2, 1, pcm(320))
	waitFor(t, "transcribe call", func() bool { return len(svc.Calls()) == 1 })
	c.send(typeDescribe, nil, nil)
	if ev := c.recv(); ev.Type != typeInfo {
		t.Fatalf("got %s before info", ev.Type)
	}
	close(release)
	if ev := c.recv(); ev.Type != typeTranscript {
		t.Fatalf("got %s", ev.Type)
	}
}

func TestHalfCloseAfterAudioStopStillAnswers(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{StreamTranscripts: true}, svc, nil)
	c := h.dial()
	c.utterance(16000, 2, 1, pcm(320))
	if err := c.nc.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}
	var types []string
	for len(types) == 0 || types[len(types)-1] != typeTranscriptStop {
		types = append(types, c.recv().Type)
	}
	if !slices.Contains(types, typeTranscript) {
		t.Errorf("no transcript after half-close: %v", types)
	}
	c.expectClosed()
}

func TestMaxConnections(t *testing.T) {
	h := startServer(t, Config{MaxConnections: 1}, newFakeService(), nil)
	first := h.dial()
	first.send(typePing, nil, nil)
	first.recv() // the first connection is registered before dialing again
	second := h.dial()
	ev := second.recv()
	if ev.Type != typeError || decode[errorData](t, ev).Code != "unavailable" {
		t.Fatalf("got %s %v", ev.Type, ev.Data)
	}
	second.expectClosed()
	_ = first.nc.Close()
	waitFor(t, "slot freed", func() bool {
		c := h.dial()
		c.send(typePing, nil, nil)
		return c.recv().Type == typePong
	})
}

func TestDisconnectMidUtterance(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	c.send(typeAudioStart, map[string]any{"rate": 16000, "width": 2, "channels": 1}, nil)
	c.send(typeAudioChunk, map[string]any{"rate": 16000, "width": 2, "channels": 1}, pcm(320))
	_ = c.nc.Close()
	// A fresh connection still works and the dropped stream never ran.
	c2 := h.dial()
	c2.utterance(16000, 2, 1, pcm(320))
	if ev := c2.recv(); ev.Type != typeTranscript {
		t.Fatalf("got %s", ev.Type)
	}
	if n := len(svc.Calls()); n != 1 {
		t.Errorf("calls %d", n)
	}
}

func TestUtteranceLimits(t *testing.T) {
	t.Run("bytes", func(t *testing.T) {
		h := startServer(t, Config{MaxUtteranceBytes: 1000}, newFakeService(), nil)
		c := h.dial()
		c.send(typeAudioStart, map[string]any{"rate": 16000, "width": 2, "channels": 1}, nil)
		c.send(typeAudioChunk, map[string]any{"rate": 16000, "width": 2, "channels": 1}, pcm(600))
		c.send(typeAudioChunk, map[string]any{"rate": 16000, "width": 2, "channels": 1}, pcm(600))
		if ev := c.recv(); ev.Type != typeError {
			t.Fatalf("got %s", ev.Type)
		}
		c.expectClosed()
	})
	t.Run("duration", func(t *testing.T) {
		h := startServer(t, Config{MaxUtteranceDuration: 50 * time.Millisecond}, newFakeService(), nil)
		c := h.dial()
		c.send(typeAudioStart, map[string]any{"rate": 16000, "width": 2, "channels": 1}, nil)
		c.send(typeAudioChunk, map[string]any{"rate": 16000, "width": 2, "channels": 1}, pcm(3200)) // 100 ms
		if ev := c.recv(); ev.Type != typeError {
			t.Fatalf("got %s", ev.Type)
		}
		c.expectClosed()
	})
	t.Run("frame", func(t *testing.T) {
		h := startServer(t, Config{MaxPayloadBytes: 100}, newFakeService(), nil)
		c := h.dial()
		c.send(typeAudioChunk, map[string]any{"rate": 16000, "width": 2, "channels": 1}, pcm(101))
		ev := c.recv()
		if ev.Type != typeError || decode[errorData](t, ev).Code != "bad_request" {
			t.Fatalf("got %s %v", ev.Type, ev.Data)
		}
		c.expectClosed()
	})
	t.Run("garbage", func(t *testing.T) {
		h := startServer(t, Config{}, newFakeService(), nil)
		c := h.dial()
		_, _ = c.nc.Write([]byte("GET / HTTP/1.1\r\n\r\n"))
		if ev := c.recv(); ev.Type != typeError {
			t.Fatalf("got %s", ev.Type)
		}
		c.expectClosed()
	})
}

func TestIdleTimeout(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{IdleTimeout: 100 * time.Millisecond}, svc, nil)
	c := h.dial()
	start := time.Now()
	c.expectClosed()
	if time.Since(start) > 2*time.Second {
		t.Errorf("closed after %v", time.Since(start))
	}

	// A pending transcription keeps the connection alive past the timeout.
	svc.fn = func(ctx context.Context, req *core.Request) (*core.Result, error) {
		time.Sleep(300 * time.Millisecond)
		return &core.Result{Text: "slow"}, nil
	}
	c2 := h.dial()
	c2.utterance(16000, 2, 1, pcm(320))
	if ev := c2.recv(); ev.Type != typeTranscript {
		t.Fatalf("got %s", ev.Type)
	}
}

func TestServiceErrorMapping(t *testing.T) {
	svc := newFakeService()
	svc.fn = func(ctx context.Context, req *core.Request) (*core.Result, error) {
		return nil, core.WrapError(core.KindForbidden, errors.New("secret /etc/users.yaml"), "model not allowed")
	}
	h := startServer(t, Config{StreamTranscripts: true}, svc, nil)
	c := h.dial()
	c.utterance(16000, 2, 1, pcm(320))
	ev := c.recv()
	if ev.Type != typeError {
		t.Fatalf("got %s", ev.Type)
	}
	e := decode[errorData](t, ev)
	if e.Code != "forbidden" || e.Text != "model not allowed" {
		t.Errorf("error %+v", e)
	}
	c.expectClosed()
}

func TestDecoderErrorMapping(t *testing.T) {
	h := startServer(t, Config{}, newFakeService(), nil)
	h.dec.err = core.Errorf(core.KindBadAudio, "cannot convert audio")
	c := h.dial()
	c.utterance(16000, 2, 1, pcm(320))
	ev := c.recv()
	if ev.Type != typeError || decode[errorData](t, ev).Code != "bad_audio" {
		t.Fatalf("got %s %v", ev.Type, ev.Data)
	}
	c.expectClosed()
}

func TestIdentify(t *testing.T) {
	svc := newFakeService()
	var seen atomic.Value
	h := startServer(t, Config{}, svc, func(remote net.Addr) (*core.Identity, error) {
		seen.Store(remote.String())
		if strings.HasPrefix(remote.String(), "127.0.0.1:") {
			return nil, core.Errorf(core.KindForbidden, "address not allowed")
		}
		return &core.Identity{User: "x"}, nil
	})
	c := h.dial()
	ev := c.recv()
	e := decode[errorData](t, ev)
	if ev.Type != typeError || e.Code != "forbidden" || e.Text != "address not allowed" {
		t.Fatalf("got %s %+v", ev.Type, e)
	}
	c.expectClosed()
	if seen.Load() == nil {
		t.Error("identify not called")
	}
}

func TestIdentityInModelsContext(t *testing.T) {
	svc := newFakeService()
	id := &core.Identity{User: "hass"}
	h := startServer(t, Config{}, svc, func(net.Addr) (*core.Identity, error) { return id, nil })
	c := h.dial()
	c.send(typeDescribe, nil, nil)
	c.recv()
	svc.mu.Lock()
	defer svc.mu.Unlock()
	if len(svc.modelIDs) != 1 || svc.modelIDs[0] != id {
		t.Errorf("Models ctx identity: %+v", svc.modelIDs)
	}
}

func TestNetworkAdminWhenNoIdentify(t *testing.T) {
	svc := newFakeService()
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	c.utterance(16000, 2, 1, pcm(320))
	c.recv()
	if id := svc.Calls()[0].id; !id.NetworkAdmin {
		t.Errorf("identity %+v", id)
	}
}

func TestPingAndUnknownEvents(t *testing.T) {
	h := startServer(t, Config{}, newFakeService(), nil)
	c := h.dial()
	c.send("run-satellite", nil, nil)
	c.send(typeSelectProgram, map[string]any{"name": "other"}, nil)
	c.send(typePing, map[string]any{"text": "hi"}, nil)
	ev := c.recv()
	if ev.Type != typePong || *decode[pingData](t, ev).Text != "hi" {
		t.Fatalf("got %s %v", ev.Type, ev.Data)
	}
}

func TestGracefulShutdown(t *testing.T) {
	t.Run("running job finishes", func(t *testing.T) {
		svc := newFakeService()
		release := make(chan struct{})
		svc.fn = func(ctx context.Context, req *core.Request) (*core.Result, error) {
			select {
			case <-release:
				return &core.Result{Text: "finished"}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		h := startServer(t, Config{ShutdownGrace: 5 * time.Second}, svc, nil)
		busy := h.dial()
		idle := h.dial()
		idle.send(typePing, nil, nil)
		idle.recv()
		busy.utterance(16000, 2, 1, pcm(320))
		waitFor(t, "transcribe call", func() bool { return len(svc.Calls()) == 1 })

		h.cancel()
		idle.expectClosed()
		waitFor(t, "listener close", func() bool {
			nc, err := net.DialTimeout("tcp", h.addr, time.Second)
			if err != nil {
				return true
			}
			_ = nc.Close()
			return false
		})
		close(release)
		ev := busy.recv()
		if ev.Type != typeTranscript || decode[transcriptData](t, ev).Text != "finished" {
			t.Fatalf("got %s %v", ev.Type, ev.Data)
		}
		busy.expectClosed()
		select {
		case err := <-h.done:
			if err != nil {
				t.Errorf("Serve: %v", err)
			}
			h.done <- err
		case <-time.After(5 * time.Second):
			t.Fatal("Serve did not return")
		}
	})
	t.Run("grace expires", func(t *testing.T) {
		svc := newFakeService()
		svc.fn = func(ctx context.Context, req *core.Request) (*core.Result, error) {
			<-ctx.Done()
			return nil, core.WrapError(core.KindCanceled, ctx.Err(), "canceled")
		}
		h := startServer(t, Config{ShutdownGrace: 50 * time.Millisecond}, svc, nil)
		c := h.dial()
		c.utterance(16000, 2, 1, pcm(320))
		waitFor(t, "transcribe call", func() bool { return len(svc.Calls()) == 1 })
		h.cancel()
		ev := c.recv()
		e := decode[errorData](t, ev)
		if ev.Type != typeError || e.Code != "unavailable" || e.Text != msgShutdown {
			t.Fatalf("got %s %+v", ev.Type, e)
		}
		c.expectClosed()
	})
	t.Run("recording gets error", func(t *testing.T) {
		h := startServer(t, Config{}, newFakeService(), nil)
		c := h.dial()
		c.send(typeAudioStart, map[string]any{"rate": 16000, "width": 2, "channels": 1}, nil)
		c.send(typePing, nil, nil)
		c.recv()
		h.cancel()
		if ev := c.recv(); ev.Type != typeError {
			t.Fatalf("got %s", ev.Type)
		}
		c.expectClosed()
	})
}

func TestLanguageHandling(t *testing.T) {
	svc := newFakeService()
	svc.models = []core.Model{
		{ID: "tiny", Aliases: []string{"whisper-1"}, Languages: []string{"en", "pt", "yue", "zh", "nl"}, Default: true},
		{ID: "small", Languages: []string{"en", "de"}},
		{ID: "bare"}, // no list: all Whisper languages
	}
	h := startServer(t, Config{}, svc, nil)
	cases := []struct{ model, lang, want string }{
		{"", "", ""},
		{"", "auto", ""},
		{"", "AUTO", ""},
		{"", "EN", "en"},
		{"", "en-US", "en"},
		{"", "en_GB", "en"},
		{"", "pt-BR", "pt"},
		{"", "zh-HK", "yue"},
		{"", "zh_hk", "yue"},
		{"", "zh-TW", "zh"},
		{"", "ga", ""},  // HA language Whisper lacks: auto-detect
		{"", "de", ""},  // not offered by the default model
		{"", "fil", ""}, // never a prefix match onto "fi"
		{"small", "de-AT", "de"},
		{"small", "zh-HK", ""},
		{"whisper-1", "nl", "nl"},
		{"bare", "de-AT", "de"},
		{"unknown", "nl", "nl"}, // falls back to the default model's list
	}
	for i, tc := range cases {
		c := h.dial()
		c.send(typeTranscribe, map[string]any{"name": tc.model, "language": tc.lang}, nil)
		c.utterance(16000, 2, 1, pcm(320))
		if ev := c.recv(); ev.Type != typeTranscript {
			t.Fatalf("%+v: got %s", tc, ev.Type)
		}
		req := svc.Calls()[i].req
		if req.Language != tc.want || req.Model != tc.model {
			t.Errorf("model %q language %q -> request model %q language %q, want language %q",
				tc.model, tc.lang, req.Model, req.Language, tc.want)
		}
	}
}

func TestAliasesNotListedAsModels(t *testing.T) {
	svc := newFakeService()
	svc.models[0].Aliases = []string{"whisper-1"}
	h := startServer(t, Config{}, svc, nil)
	c := h.dial()
	c.send(typeDescribe, nil, nil)
	for _, m := range decode[infoData](t, c.recv()).ASR[0].Models {
		if m.Name == "whisper-1" {
			t.Error("alias listed as a model")
		}
	}
}

func TestConfigValidation(t *testing.T) {
	if _, err := New(Config{MaxPayloadBytes: -1}, newFakeService(), &fakeDecoder{}, nil, nil); err == nil {
		t.Error("negative byte limit accepted")
	}
	if _, err := New(Config{}, nil, &fakeDecoder{}, nil, nil); err == nil {
		t.Error("nil Service accepted")
	}
}

func TestServeReturnsOnListenerClose(t *testing.T) {
	srv, err := New(Config{}, newFakeService(), &fakeDecoder{}, nil, testLogger())
	if err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- srv.Serve(context.Background(), ln) }()
	time.Sleep(20 * time.Millisecond)
	_ = ln.Close()
	select {
	case err := <-done:
		if err == nil {
			t.Error("want an error when the listener is closed underneath Serve")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Serve did not return")
	}
}
