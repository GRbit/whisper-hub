package audio

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/GRbit/whisper-hub/internal/core"
)

// Accepted FromRaw parameter ranges; anything outside is not real audio.
const (
	minRate     = 1000
	maxRate     = 384000
	maxChannels = 32
)

// FromRaw converts headerless little-endian PCM to mono 16 kHz s16le.
// Width 1 is unsigned 8-bit (as Wyoming and WAV define it), widths 2, 3 and
// 4 are signed integers. Channels are averaged; other rates go through a
// band-limited windowed-sinc resampler. 16 kHz mono 16-bit input is returned
// as is, without copying.
func (d *Decoder) FromRaw(ctx context.Context, data []byte, rate, width, channels int) (core.PCM, error) {
	if width < 1 || width > 4 || channels < 1 || channels > maxChannels || rate < minRate || rate > maxRate {
		return nil, core.Errorf(core.KindBadAudio, "unsupported raw audio format: rate %d, width %d, channels %d",
			rate, width, channels)
	}
	if rate == core.SampleRate && width == 2 && channels == 1 {
		// Same checks and texts as upstream's encode=false path.
		var reason string
		switch {
		case len(data)%2 != 0:
			reason = fmt.Sprintf("Raw PCM payload has an odd length (%d bytes)", len(data))
		case len(data) == 0:
			reason = "Audio decoded to 0 samples"
		default:
			return core.PCM(data), nil
		}
		return nil, core.Errorf(core.KindBadAudio, "%s", clientDetail(reason, int64(len(data)), data[:min(magicLen, len(data))]))
	}
	frame := width * channels
	if len(data)%frame != 0 {
		return nil, core.Errorf(core.KindBadAudio, "raw audio length %d is not a multiple of the frame size %d", len(data), frame)
	}
	if len(data) == 0 {
		return nil, core.Errorf(core.KindBadAudio, "raw audio is empty")
	}
	mono := downmix(data, width, channels)
	if rate != core.SampleRate {
		var err error
		if mono, err = resample(ctx, mono, rate, core.SampleRate); err != nil {
			return nil, core.WrapError(core.KindCanceled, err, "request canceled")
		}
	}
	out := make(core.PCM, 2*len(mono))
	for i, v := range mono {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(toInt16(v)))
	}
	d.log.Debug("converted raw audio", "rate", rate, "width", width, "channels", channels,
		"in_bytes", len(data), "out_samples", len(mono))
	return out, nil
}

func toInt16(v float32) int16 {
	s := math.Round(float64(v) * 32768)
	return int16(max(-32768, min(32767, s)))
}

// downmix decodes frames to float32 in [-1, 1) and averages channels.
func downmix(data []byte, width, channels int) []float32 {
	n := len(data) / (width * channels)
	out := make([]float32, n)
	scale := 1 / float32(channels)
	p := 0
	for i := range out {
		var acc float32
		for range channels {
			acc += sampleAt(data[p:], width)
			p += width
		}
		out[i] = acc * scale
	}
	return out
}

func sampleAt(b []byte, width int) float32 {
	switch width {
	case 1:
		return float32(int(b[0])-128) / 128
	case 2:
		return float32(int16(binary.LittleEndian.Uint16(b))) / 32768
	case 3:
		v := int32(uint32(b[0])|uint32(b[1])<<8|uint32(b[2])<<16) << 8 >> 8
		return float32(v) / (1 << 23)
	default:
		return float32(float64(int32(binary.LittleEndian.Uint32(b))) / (1 << 31))
	}
}

// Resampler parameters. 16 zero crossings per side with a Kaiser window
// (beta 8.6) gives roughly 80 dB stopband rejection; the cutoff sits at 94%
// of the lower Nyquist frequency so the transition band ends before it.
const (
	kernelZeros      = 16
	kernelOversample = 512
	kaiserBeta       = 8.6
	cutoffRolloff    = 0.94
)

var (
	kernelOnce  sync.Once
	kernelTable []float32
)

// kernel returns sinc(u) * kaiser(u / kernelZeros) for u in [0, kernelZeros],
// sampled kernelOversample times per unit and linearly interpolated. A table
// keeps math.Sin out of the inner loop.
func kernel(u float64) float32 {
	kernelOnce.Do(func() {
		n := kernelZeros*kernelOversample + 2
		kernelTable = make([]float32, n)
		i0b := besselI0(kaiserBeta)
		for i := range kernelTable {
			x := float64(i) / kernelOversample
			if x > kernelZeros {
				continue
			}
			s := 1.0
			if x != 0 {
				s = math.Sin(math.Pi*x) / (math.Pi * x)
			}
			r := x / kernelZeros
			w := besselI0(kaiserBeta*math.Sqrt(1-r*r)) / i0b
			kernelTable[i] = float32(s * w)
		}
	})
	pos := u * kernelOversample
	i := int(pos)
	if i >= len(kernelTable)-1 {
		return 0
	}
	f := float32(pos - float64(i))
	return kernelTable[i] + f*(kernelTable[i+1]-kernelTable[i])
}

func besselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 50; k++ {
		term *= (x / (2 * float64(k))) * (x / (2 * float64(k)))
		sum += term
		if term < 1e-12*sum {
			break
		}
	}
	return sum
}

// resample converts in from inRate to outRate with a windowed-sinc low-pass
// whose cutoff follows the lower of the two Nyquist frequencies, so
// downsampling does not alias and upsampling does not image.
func resample(ctx context.Context, in []float32, inRate, outRate int) ([]float32, error) {
	nOut := int(int64(len(in)) * int64(outRate) / int64(inRate))
	out := make([]float32, nOut)
	step := float64(inRate) / float64(outRate)
	fc := math.Min(1, float64(outRate)/float64(inRate)) * cutoffRolloff
	support := kernelZeros / fc // in input samples
	gain := float32(fc)
	for n := range out {
		if n&0xffff == 0 && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		x := float64(n) * step
		lo := max(0, int(math.Ceil(x-support)))
		hi := min(len(in)-1, int(math.Floor(x+support)))
		var acc float32
		for k := lo; k <= hi; k++ {
			acc += in[k] * kernel(math.Abs(x-float64(k))*fc)
		}
		out[n] = acc * gain
	}
	return out, nil
}
