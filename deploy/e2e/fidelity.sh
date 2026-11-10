#!/usr/bin/env bash
# Legacy fidelity through the whole stack: the reference whisper-asr-webservice
# image (run as internal/legacy/testdata/reference/capture.sh does: upstream's
# main app over whisper-asr-webservice:ec32cd5, model tiny, int8) against a
# whisper-hub all-in-one image, same original uploads, encode=true.
#   ENGINE=whisperx deploy/e2e/fidelity.sh            image whisper-hub:whisperx
#   ENGINE=all ASR_ENGINE=faster_whisper deploy/e2e/fidelity.sh
#   REF_TWICE=1 ...   first compare the reference with itself, so cases on
#                     which upstream is not deterministic are labelled UNSTABLE
# Extra arguments go to fidelity.py (e.g. --cases default).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ref_repo="${REF_REPO:-/path/to/whisper-asr-webservice}"
ref_image="${REF_IMAGE:-whisper-asr-webservice:ec32cd5}"
image_engine="${ENGINE:-faster_whisper}"
aio_image="${AIO_IMAGE:-whisper-hub:$image_engine}"
# The engine under test: ASR_ENGINE, else the image's (all -> upstream's default).
engine="${ASR_ENGINE:-$image_engine}"
[ "$engine" = all ] && engine=openai_whisper
cache_volume="${CACHE_VOLUME:-whub-e2e-cache}"
voice="${VOICE_FILES:-/path/to/voice-test-files}"
work="$(mktemp -d)"
cleanup() { docker rm -f whub-fid-ref whub-fid-hub >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT

mkdir -p "$work/audio"
cp "$voice"/*.wav "$work/audio/"
ffmpeg -v error -i "$voice/german-20s.wav" -c:a aac -b:a 96k "$work/audio/german-trailing-moov.m4a"
ffmpeg -v error -i "$voice/german-20s.wav" -c:a libmp3lame -b:a 64k "$work/audio/german.mp3"
ffmpeg -v error -i "$voice/english-front-center.wav" -c:a libopus "$work/audio/english.ogg"
(cd "$ref_repo" && git archive main app) | tar -x -C "$work"

model=(-e ASR_MODEL=tiny -e ASR_QUANTIZATION=int8 -v "$cache_volume:/root/.cache")
ref_extra=()
if [ "$engine" = whisperx ]; then
	# Test-only: upstream's whisperx cannot load its bundled VAD checkpoint
	# under torch 2.6+; this is the same narrow allowlist the worker uses.
	ref_extra=(-v "$here/reference/sitecustomize.py:/refpatch/sitecustomize.py:ro" -e PYTHONPATH=/refpatch)
fi
hub_env=()
[ -n "${ASR_ENGINE:-}" ] && hub_env=(-e "ASR_ENGINE=$ASR_ENGINE")
docker run -d --name whub-fid-ref -p 127.0.0.1:19010:9000 -w /app -v "$work/app:/app/app:ro" \
	-e "ASR_ENGINE=$engine" "${model[@]}" "${ref_extra[@]}" "$ref_image" >/dev/null
docker run -d --name whub-fid-hub -p 127.0.0.1:19020:9000 "${model[@]}" "${hub_env[@]}" "$aio_image" >/dev/null
for u in 19010/openapi.json 19020/healthz; do
	for _ in $(seq 600); do curl -sf "http://127.0.0.1:$u" >/dev/null && break; sleep 2; done
done
echo "== fidelity: $aio_image ${ASR_ENGINE:+ASR_ENGINE=$ASR_ENGINE }vs $ref_image ASR_ENGINE=$engine"
args=(--engine "$engine")
if [ -n "${REF_TWICE:-}" ]; then
	echo "-- reference vs itself"
	python3 "$here/fidelity.py" --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19010 --engine none \
		--write-diffs "$work/unstable.txt" "$@" "$work"/audio/* | tail -1 || true
	args+=(--unstable "$work/unstable.txt")
	echo "-- reference vs whisper-hub"
fi
python3 "$here/fidelity.py" --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19020 "${args[@]}" "$@" "$work"/audio/*
