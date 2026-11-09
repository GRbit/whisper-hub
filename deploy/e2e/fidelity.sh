#!/usr/bin/env bash
# Legacy fidelity through the whole stack: the reference whisper-asr-webservice
# image (run as internal/legacy/testdata/reference/capture.sh does: upstream's
# main app over whisper-asr-webservice:ec32cd5, faster_whisper, tiny, int8)
# against the whisper-hub all-in-one image, with the same original uploads
# and encode=true. See fidelity.py for what is compared.
#   deploy/e2e/fidelity.sh [fidelity.py args, e.g. --cases default]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ref_repo="${REF_REPO:-/path/to/whisper-asr-webservice}"
ref_image="${REF_IMAGE:-whisper-asr-webservice:ec32cd5}"
aio_image="${AIO_IMAGE:-whisper-hub:faster_whisper}"
voice="${VOICE_FILES:-/path/to/voice-test-files}"
work="$(mktemp -d)"
cleanup() { docker rm -f whub-fid-ref whub-fid-hub >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT

mkdir -p "$work/audio" "$work/models"
cp "$voice"/*.wav "$work/audio/"
ffmpeg -v error -i "$voice/german-20s.wav" -c:a aac -b:a 96k "$work/audio/german-trailing-moov.m4a"
ffmpeg -v error -i "$voice/german-20s.wav" -c:a libmp3lame -b:a 64k "$work/audio/german.mp3"
ffmpeg -v error -i "$voice/english-front-center.wav" -c:a libopus "$work/audio/english.ogg"
(cd "$ref_repo" && git archive main app) | tar -x -C "$work"

model=(-e ASR_MODEL=tiny -e ASR_MODEL_PATH=/models -e ASR_QUANTIZATION=int8)
docker run -d --name whub-fid-ref -p 127.0.0.1:19010:9000 -w /app -v "$work/app:/app/app:ro" -v "$work/models:/models" \
	-e ASR_ENGINE=faster_whisper "${model[@]}" "$ref_image" >/dev/null
docker run -d --name whub-fid-hub -p 127.0.0.1:19020:9000 -v "$work/models:/models" "${model[@]}" "$aio_image" >/dev/null
for u in 19010/openapi.json 19020/healthz; do
	for _ in $(seq 300); do curl -sf "http://127.0.0.1:$u" >/dev/null && break; sleep 2; done
done
python3 "$here/fidelity.py" --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19020 "$@" "$work"/audio/*
