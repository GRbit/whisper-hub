package workerclient_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/core"
	"github.com/GRbit/whisper-hub/internal/hub/internal/workerclient"
	"github.com/GRbit/whisper-hub/internal/hub/workertest"
)

func newClient(t *testing.T, url string, cfg workerclient.Config) *workerclient.Client {
	t.Helper()
	c, err := workerclient.New(url, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

func pcm(seconds float64) core.PCM { return make(core.PCM, int(seconds*core.SampleRate)*2) }

func wantKind(t *testing.T, err error, kind core.ErrorKind) {
	t.Helper()
	if err == nil {
		t.Fatalf("want %s error, got nil", kind)
	}
	if got := core.KindOf(err); got != kind {
		t.Fatalf("want kind %s, got %s (%v)", kind, got, err)
	}
}

func TestInfoAndHealth(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	c := newClient(t, w.URL, workerclient.Config{})
	if err := c.Health(context.Background()); err != nil {
		t.Fatal(err)
	}
	info, err := c.Info(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if info.Engine != "faster_whisper" || info.DefaultModel() != "tiny" || !info.Capabilities.Translate {
		t.Fatalf("unexpected info %+v", info)
	}
	w.SetHealthy(false)
	wantKind(t, c.Health(context.Background()), core.KindUnavailable)
}

func TestTranscribeStreamsSegmentsAndSendsOptions(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	c := newClient(t, w.URL, workerclient.Config{})
	req := &core.Request{Language: "de", WordTimestamps: true, LegacyOutput: "srt"}
	audio := pcm(2.5)
	var got []int
	res, err := c.Transcribe(context.Background(), workerclient.OptionsFromRequest("tiny", req), audio,
		func(s core.Segment) error { got = append(got, s.ID); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || len(res.Segments) != 3 || res.Text != " s0 s1 s2" || res.Language != "de" {
		t.Fatalf("segments %v result %+v", got, res)
	}
	if string(res.Rendered) != "srt: s0 s1 s2" || res.Model != "tiny" || res.Engine != "faster_whisper" {
		t.Fatalf("result %+v", res)
	}
	job := w.Jobs()[0]
	if len(job.Audio) != len(audio) {
		t.Fatalf("worker got %d audio bytes, want %d", len(job.Audio), len(audio))
	}
	raw := string(job.RawOptions)
	for _, want := range []string{`"task":"transcribe"`, `"language":"de"`, `"prompt":null`, `"temperature":null`,
		`"word_timestamps":true`, `"min_speakers":null`, `"legacy_output":"srt"`} {
		if !strings.Contains(raw, want) {
			t.Errorf("options %s lack %s", raw, want)
		}
	}
}

// Segments must reach the callback while the worker is still producing,
// not after the stream ends.
func TestTranscribeDeliversSegmentsBeforeDone(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	release := make(chan struct{})
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Info("tiny", "faster_whisper", "en", 1)
		s.Segment(core.Segment{ID: 0, Text: " a"})
		<-release
		s.Done(" a", "en", nil)
	})
	c := newClient(t, w.URL, workerclient.Config{})
	done := make(chan error, 1)
	seen := make(chan struct{})
	go func() {
		_, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), func(core.Segment) error {
			close(seen)
			return nil
		})
		done <- err
	}()
	select {
	case <-seen:
	case <-time.After(5 * time.Second):
		t.Fatal("segment not delivered while stream open")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestTranscribeToleratesNullsAndUnknownFields(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Raw(`{"type":"info","model":"tiny","engine":"x","language":"en","extra":{"a":1}}`)
		s.Raw(`{"type":"segment","segment":{"id":0,"start":0,"end":1,"text":" hi","tokens":null,"words":null,"speaker":null,"new_field":[1]}}`)
		s.Raw(`{"type":"segment","segment":{"id":1,"start":1,"end":2,"text":" yo","words":[{"word":" yo","start":1,"end":2,"probability":0.5,"speaker":null}]}}`)
		s.Raw(`{"type":"future_event","x":1}`)
		s.Raw(`{"type":"done","text":" hi yo","language":"en","rendered":null}`)
	})
	c := newClient(t, w.URL, workerclient.Config{})
	res, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Segments) != 2 || res.Segments[0].Words != nil || res.Segments[1].Words[0].Word != " yo" || res.Rendered != nil {
		t.Fatalf("result %+v", res)
	}
}

func TestTranscribeErrors(t *testing.T) {
	cases := []struct {
		name   string
		script workertest.TranscribeFunc
		kind   core.ErrorKind
	}{
		{"http bad request", func(_ *workertest.Job, s *workertest.Stream) { s.HTTPError("bad_request", "language xx") }, core.KindBadRequest},
		{"http model not found", func(_ *workertest.Job, s *workertest.Stream) { s.HTTPError("model_not_found", "nope") }, core.KindModelNotFound},
		{"http unavailable", func(_ *workertest.Job, s *workertest.Stream) { s.HTTPError("unavailable", "busy") }, core.KindUnavailable},
		{"http internal", func(_ *workertest.Job, s *workertest.Stream) { s.HTTPError("internal", "Traceback /opt/x.py") }, core.KindInternal},
		{"stream error", func(_ *workertest.Job, s *workertest.Stream) {
			s.Info("tiny", "x", "en", 1)
			s.Error("internal", "CUDA OOM")
		}, core.KindInternal},
		{"truncated stream", func(_ *workertest.Job, s *workertest.Stream) { s.Info("tiny", "x", "en", 1) }, core.KindInternal},
		{"garbage line", func(_ *workertest.Job, s *workertest.Stream) { s.Raw("not json") }, core.KindInternal},
	}
	w := workertest.New(t, workertest.DefaultInfo())
	c := newClient(t, w.URL, workerclient.Config{})
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w.SetTranscribe(tc.script)
			_, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
			wantKind(t, err, tc.kind)
			if strings.Contains(core.ClientMessage(err), "Traceback") || strings.Contains(core.ClientMessage(err), "CUDA") {
				t.Fatalf("worker internals leaked into client message %q", core.ClientMessage(err))
			}
		})
	}
}

func TestPlainHTTPErrorWithoutJSONBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "<html>bad gateway</html>", http.StatusNotFound)
	}))
	defer srv.Close()
	c := newClient(t, srv.URL, workerclient.Config{})
	_, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	wantKind(t, err, core.KindModelNotFound)
}

func TestTranscribeConnectionRefusedIsUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	c := newClient(t, url, workerclient.Config{})
	_, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	wantKind(t, err, core.KindUnavailable)
	wantKind(t, c.Health(context.Background()), core.KindUnavailable)
}

// Canceling the caller context must abort the HTTP request so the worker
// sees the disconnect (its request context ends) and can stop the job.
func TestCancelPropagatesToWorker(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	workerSawCancel := make(chan struct{})
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Info("tiny", "x", "en", 1)
		<-job.Ctx.Done()
		close(workerSawCancel)
	})
	c := newClient(t, w.URL, workerclient.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := c.Transcribe(ctx, workerclient.Options{}, pcm(1), nil)
		errc <- err
	}()
	<-w.Started()
	cancel()
	wantKind(t, <-errc, core.KindCanceled)
	select {
	case <-workerSawCancel:
	case <-time.After(5 * time.Second):
		t.Fatal("worker request context was not canceled")
	}
}

func TestSegmentCallbackErrorAbortsJob(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	workerSawCancel := make(chan struct{})
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Info("tiny", "x", "en", 1)
		s.Segment(core.Segment{ID: 0})
		<-job.Ctx.Done()
		close(workerSawCancel)
	})
	c := newClient(t, w.URL, workerclient.Config{})
	boom := errors.New("client write failed")
	_, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), func(core.Segment) error { return boom })
	wantKind(t, err, core.KindCanceled)
	if !errors.Is(err, boom) {
		t.Fatalf("cause lost: %v", err)
	}
	select {
	case <-workerSawCancel:
	case <-time.After(5 * time.Second):
		t.Fatal("worker request context was not canceled")
	}
}

func TestFirstByteTimeout(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) { <-job.Ctx.Done() })
	c := newClient(t, w.URL, workerclient.Config{FirstByteTimeout: 50 * time.Millisecond})
	_, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	wantKind(t, err, core.KindUnavailable)
}

// Once the stream has started, a long silence (non-streaming engines emit
// everything at the end) must not trip the first-byte timeout.
func TestNoTimeoutAfterFirstByte(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Info("tiny", "x", "en", 1)
		time.Sleep(200 * time.Millisecond) // > FirstByteTimeout below
		s.Done("", "en", nil)
	})
	c := newClient(t, w.URL, workerclient.Config{FirstByteTimeout: 50 * time.Millisecond})
	if _, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil); err != nil {
		t.Fatal(err)
	}
}

func TestLongLine(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	big := strings.Repeat("x", 300<<10) // larger than the 64 KiB read buffer
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Info("tiny", "x", "en", 1)
		s.Done(big, "en", nil)
	})
	c := newClient(t, w.URL, workerclient.Config{})
	res, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	if err != nil || res.Text != big {
		t.Fatalf("err %v len %d", err, len(res.Text))
	}
	c2 := newClient(t, w.URL, workerclient.Config{MaxLineBytes: 100 << 10})
	_, err = c2.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	wantKind(t, err, core.KindInternal)
}

func TestDetectLanguage(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetDetect(func(job *workertest.Job) (string, float64, string) { return "de", 0.7, "" })
	c := newClient(t, w.URL, workerclient.Config{})
	d, err := c.DetectLanguage(context.Background(), "", pcm(1))
	if err != nil {
		t.Fatal(err)
	}
	if d.Language != "de" || d.Confidence != 0.7 || d.Model != "tiny" || d.Engine != "faster_whisper" {
		t.Fatalf("detection %+v", d)
	}
	if got := string(w.Jobs()[0].RawOptions); got != `{"model":null}` {
		t.Fatalf("options %s", got)
	}
	w.SetDetect(func(job *workertest.Job) (string, float64, string) { return "", 0, "model_not_found" })
	_, err = c.DetectLanguage(context.Background(), "nope", pcm(1))
	wantKind(t, err, core.KindModelNotFound)
}

func TestNewRejectsBadURL(t *testing.T) {
	for _, u := range []string{"", "worker:9100", "ftp://x", "http://"} {
		if _, err := workerclient.New(u, workerclient.Config{}, nil); err == nil {
			t.Errorf("url %q accepted", u)
		}
	}
}

// Python's json.dumps writes float("nan") and float("-inf") as bare
// tokens, which are not JSON; they must not fail the job.
func TestTranscribeToleratesPythonNonFiniteNumbers(t *testing.T) {
	w := workertest.New(t, workertest.DefaultInfo())
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) {
		s.Raw(`{"type":"info","model":"tiny","engine":"x","language":"en","language_probability":NaN}`)
		s.Raw(`{"type":"segment","segment":{"id":0,"text":" NaN -Infinity \" Infinity","avg_logprob":-Infinity,"no_speech_prob":Infinity,"compression_ratio":NaN}}`)
		s.Raw(`{"type":"done","text":" NaN","language":"en"}`)
	})
	c := newClient(t, w.URL, workerclient.Config{})
	res, err := c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	if err != nil {
		t.Fatal(err)
	}
	seg := res.Segments[0]
	if seg.Text != ` NaN -Infinity " Infinity` || seg.AvgLogprob != 0 || seg.NoSpeechProb != 0 || res.LanguageProbability != 0 {
		t.Fatalf("segment %+v result %+v", seg, res)
	}
	w.SetTranscribe(func(job *workertest.Job, s *workertest.Stream) { s.Raw(`{"type":"info",NaNx}`) })
	_, err = c.Transcribe(context.Background(), workerclient.Options{}, pcm(1), nil)
	wantKind(t, err, core.KindInternal)
}
