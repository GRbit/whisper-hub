package main

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"testing"
	"time"

	"github.com/GRbit/whisper-hub/internal/hub/workertest"
)

// byModel runs the script of the job's model.
func byModel(scripts map[string]workertest.TranscribeFunc) workertest.TranscribeFunc {
	return func(job *workertest.Job, s *workertest.Stream) { scripts[job.Model](job, s) }
}

func legacyASR(base string, audio []byte) (int, string) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("audio_file", "a.wav")
	_, _ = fw.Write(audio)
	_ = mw.Close()
	resp, err := http.Post(base+"/asr?output=txt", mw.FormDataContentType(), &body)
	if err != nil {
		return 0, err.Error()
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func openaiCode(base, model string, audio []byte) int {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "a.wav")
	_, _ = fw.Write(audio)
	_ = mw.WriteField("model", model)
	_ = mw.Close()
	resp, err := http.Post(base+"/v1/audio/transcriptions", mw.FormDataContentType(), &body)
	if err != nil {
		return 0
	}
	resp.Body.Close()
	return resp.StatusCode
}

// preemptedPair starts a legacy job on tiny, then an OpenAI job on base
// with a higher priority that pauses it, and waits until the OpenAI job
// runs. Results arrive on the returned channels.
func preemptedPair(t *testing.T, env map[string]string, low, high <-chan struct{}) (*running, *workertest.Job, chan int, chan int) {
	t.Helper()
	w := workertest.New(t, threeModels())
	w.SetTranscribe(byModel(map[string]workertest.TranscribeFunc{
		"tiny": workertest.Pausing(low), "base": workertest.Pausing(high),
	}))
	env["WHISPER_HUB_HUB_PRIORITIES_OPENAI"] = "60"
	env["WHISPER_HUB_HUB_PREEMPT_TIMEOUT"] = "1h"
	r := startGateway(t, w, env)
	audio := wav(t, 1)
	lowDone, highDone := make(chan int, 1), make(chan int, 1)
	go func() { code, _ := legacyASR(r.base, audio); lowDone <- code }()
	lj := <-w.Started()
	go func() { highDone <- openaiCode(r.base, "base", audio) }()
	if j := <-w.Started(); j.Model != "base" {
		t.Fatalf("started %s", j.Model)
	}
	if w.Paused(lj.ID) != 1 {
		t.Fatal("legacy job not paused")
	}
	return r, lj, lowDone, highDone
}

// Shutdown with a paused job: draining requests still get slots, so the
// paused job resumes once the job that paused it ends, and completes
// within the grace period.
func TestShutdownResumesPausedJob(t *testing.T) {
	low, high := make(chan struct{}), make(chan struct{})
	r, _, lowDone, highDone := preemptedPair(t, map[string]string{"WHISPER_HUB_HTTP_SHUTDOWN_GRACE": "10s"}, low, high)
	r.cancel()
	close(high)
	if code := <-highDone; code != http.StatusOK {
		t.Fatalf("openai job got %d", code)
	}
	close(low)
	if code := <-lowDone; code != http.StatusOK {
		t.Fatalf("paused legacy job got %d", code)
	}
	select {
	case err := <-r.done:
		r.done <- err
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("Run did not return")
	}
}

// When the grace period ends first, a paused job is cut off like a running
// one and its worker job canceled.
func TestShutdownGraceCancelsPausedJob(t *testing.T) {
	never := make(chan struct{})
	r, lj, lowDone, _ := preemptedPair(t, map[string]string{"WHISPER_HUB_HTTP_SHUTDOWN_GRACE": "200ms"}, never, never)
	r.cancel()
	select {
	case err := <-r.done:
		r.done <- err
	case <-time.After(5 * time.Second):
		t.Fatal("Run hung past the grace period")
	}
	select {
	case <-lj.Ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("paused worker job not canceled")
	}
	if code := <-lowDone; code == http.StatusOK {
		t.Fatal("cut-off legacy job reported success")
	}
}
