#!/usr/bin/env bash
# Legacy fidelity through the whole stack: the reference (public upstream
# whisper-asr-webservice at the commit pinned in reference/upstream.env,
# built with `make image-reference`, run as built) against a whisper-hub
# all-in-one image, same original uploads, encode=true, model tiny, int8.
# fidelity.py lists whisper-hub's own extensions to upstream separately.
#   VOICE_FILES=dir ENGINE=whisperx deploy/e2e/fidelity.sh       image whisper-hub:whisperx
#   VOICE_FILES=dir ENGINE=all WHISPER_HUB_ENGINE=faster_whisper deploy/e2e/fidelity.sh
#   REF_TWICE=1 ...   first compare the reference with itself, so cases on
#                     which upstream is not deterministic are labelled UNSTABLE
# VOICE_FILES: a directory with english-front-center.wav and german-20s.wav
# (see AGENTS.md); every *.wav in it is sent.
# Extra arguments go to fidelity.py (e.g. --cases default).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
user_ref_image="${REF_IMAGE:-}"
# shellcheck source=reference/upstream.env
. "$here/reference/upstream.env"
ref_image="${user_ref_image:-$REF_IMAGE}"
image_engine="${ENGINE:-faster_whisper}"
aio_image="${AIO_IMAGE:-whisper-hub:$image_engine}"
# The engine under test: WHISPER_HUB_ENGINE, else the image's (all -> upstream's default).
engine="${WHISPER_HUB_ENGINE:-$image_engine}"
[ "$engine" = all ] && engine=openai_whisper
cache_volume="${CACHE_VOLUME:-whub-e2e-cache}"
voice="${VOICE_FILES:-}"
if [ -z "$voice" ] || [ ! -f "$voice/english-front-center.wav" ] || [ ! -f "$voice/german-20s.wav" ]; then
	echo "fidelity.sh: set VOICE_FILES to a directory with english-front-center.wav and german-20s.wav" >&2
	exit 2
fi
if ! docker image inspect "$ref_image" >/dev/null 2>&1; then
	echo "fidelity.sh: reference image $ref_image not found; build it with: make image-reference" >&2
	exit 2
fi
work="$(mktemp -d)"
cleanup() { docker rm -f whub-fid-ref whub-fid-hub >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT

mkdir -p "$work/audio"
cp "$voice"/*.wav "$work/audio/"
ffmpeg -v error -i "$voice/german-20s.wav" -c:a aac -b:a 96k "$work/audio/german-trailing-moov.m4a"
ffmpeg -v error -i "$voice/german-20s.wav" -c:a libmp3lame -b:a 64k "$work/audio/german.mp3"
ffmpeg -v error -i "$voice/english-front-center.wav" -c:a libopus "$work/audio/english.ogg"

# For whisper-hub's decode extensions, with the reference image's ffmpeg
# (the same build whisper-hub ships): each upload's decoded duration as
# whisper-hub reads it (from a file), and which uploads decode to different
# audio through a pipe, as upstream reads them (MP4 with a trailing index,
# MP3 whose encoder delay is only readable from a seekable file).
REF_IMAGE_FOR_DECODE="$ref_image" python3 - "$work/durations.json" "$work/pipe-differs.txt" "$work"/audio/* <<'PY'
import hashlib, json, os, subprocess, sys
def decode(path, pipe):
    cmd = ["docker", "run", "--rm", "-i", "-v", os.path.dirname(path) + ":/a:ro", "--entrypoint", "ffmpeg",
           os.environ["REF_IMAGE_FOR_DECODE"], "-v", "error"]
    cmd += ["-i", "pipe:"] if pipe else ["-nostdin", "-i", "/a/" + os.path.basename(path)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    data = open(path, "rb").read() if pipe else None
    return subprocess.run(cmd, input=data, capture_output=True).stdout
durations, differs = {}, []
for path in sys.argv[3:]:
    name = os.path.basename(path)
    from_file, from_pipe = decode(path, False), decode(path, True)
    durations[name] = f"{len(from_file) // 2 / 16000:.3f}"
    if hashlib.sha256(from_file).digest() != hashlib.sha256(from_pipe).digest():
        differs.append(name)
        print(f"   {name}: decodes to {len(from_pipe)} bytes through a pipe, {len(from_file)} from a file", file=sys.stderr)
json.dump(durations, open(sys.argv[1], "w"))
open(sys.argv[2], "w").write(",".join(differs))
PY

# Both containers get upstream's variable names on purpose: the reference
# knows no others, and whisper-hub must accept them as a drop-in.
model=(-e ASR_MODEL=tiny -e ASR_QUANTIZATION=int8 -v "$cache_volume:/root/.cache")
ref_extra=()
if [ "$engine" = whisperx ]; then
	# Test-only, and the one addition to the reference: upstream's whisperx
	# cannot load its bundled VAD checkpoint under torch 2.6+ at all. This is
	# the same narrow torch allowlist the worker uses; it adds no behavior.
	ref_extra=(-v "$here/reference/sitecustomize.py:/refpatch/sitecustomize.py:ro" -e PYTHONPATH=/refpatch)
fi
hub_env=()
[ -n "${WHISPER_HUB_ENGINE:-}" ] && hub_env=(-e "ASR_ENGINE=$WHISPER_HUB_ENGINE")
docker run -d --name whub-fid-ref -p 127.0.0.1:19010:9000 \
	-e "ASR_ENGINE=$engine" "${model[@]}" "${ref_extra[@]}" "$ref_image" >/dev/null
docker run -d --name whub-fid-hub -p 127.0.0.1:19020:9000 "${model[@]}" "${hub_env[@]}" "$aio_image" >/dev/null
for u in 19010/openapi.json 19020/healthz; do
	for _ in $(seq 600); do curl -sf "http://127.0.0.1:$u" >/dev/null && break; sleep 2; done
done
echo "== fidelity: $aio_image ${WHISPER_HUB_ENGINE:+ASR_ENGINE=$WHISPER_HUB_ENGINE }vs $ref_image ($REF_URL at $REF_COMMIT) ASR_ENGINE=$engine"
args=(--engine "$engine")
if [ -n "${REF_TWICE:-}" ]; then
	echo "-- reference vs itself"
	python3 "$here/fidelity.py" --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19010 --engine none \
		--write-diffs "$work/unstable.txt" "$@" "$work"/audio/* | tail -1 || true
	args+=(--unstable "$work/unstable.txt")
	echo "-- reference vs whisper-hub"
fi
python3 "$here/fidelity.py" --ref http://127.0.0.1:19010 --hub http://127.0.0.1:19020 "${args[@]}" \
	--durations "$work/durations.json" --pipe-differs "$(cat "$work/pipe-differs.txt")" "$@" "$work"/audio/*
