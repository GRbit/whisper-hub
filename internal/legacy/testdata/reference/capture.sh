#!/usr/bin/env bash
# Regenerates the legacy package's ground truth from the reference service:
#   static/openapi-*.json  upstream's schema per engine (gen_openapi.py)
#   static/*.html, assets  Swagger UI pages and bundled files
#   testdata/reference/cases/*.json  real responses for reference_test.go
#
# The reference is upstream's main branch (adds X-Audio-Duration-Seconds)
# mounted over the app code of a built upstream image, run from /app like the
# upstream Dockerfile so /assets is served. Nothing is rebuilt.
#
# Env: REF_REPO (upstream checkout), REF_BRANCH, IMAGE, PORT, MODELS_DIR (cache).
# If a container named $NAME is already running it is reused and kept.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/../.." && pwd)
REF_REPO=${REF_REPO:-/path/to/whisper-asr-webservice}
REF_BRANCH=${REF_BRANCH:-main}
IMAGE=${IMAGE:-whisper-asr-webservice:ec32cd5}
PORT=${PORT:-19000}
NAME=${NAME:-wh-legacy-ref}
WORK=$(mktemp -d)
MODELS_DIR=${MODELS_DIR:-$WORK/models}
mkdir -p "$MODELS_DIR"
started=""
cleanup() {
	if [ -n "$started" ]; then docker rm -f "$NAME" >/dev/null; fi
	rm -rf "$WORK"
}
trap cleanup EXIT

(cd "$REF_REPO" && git archive "$REF_BRANCH" app) | tar -x -C "$WORK"

gen() {
	docker run --rm -w /app -e PYTHONPATH=/app -v "$WORK/app:/app/app:ro" \
		-v "$HERE/gen_openapi.py:/gen.py:ro" "$@" --entrypoint /app/.venv/bin/python "$IMAGE" /gen.py 2>/dev/null
}
gen -e ASR_ENGINE=faster_whisper >"$PKG/static/openapi-faster_whisper.json"
gen -e ASR_ENGINE=openai_whisper >"$PKG/static/openapi-openai_whisper.json"
gen -e ASR_ENGINE=whisperx >"$PKG/static/openapi-whisperx.json"
gen -e ASR_ENGINE=whisperx -e HF_TOKEN=x >"$PKG/static/openapi-whisperx_diarize.json"

for f in swagger-ui.css swagger-ui-bundle.js; do
	docker run --rm --entrypoint cat "$IMAGE" "/app/swagger-ui-assets/$f" >"$PKG/static/$f"
done

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
	docker run -d --name "$NAME" -p "127.0.0.1:$PORT:9000" -w /app \
		-v "$WORK/app:/app/app:ro" -v "$MODELS_DIR:/models" \
		-e ASR_ENGINE=faster_whisper -e ASR_MODEL=tiny -e ASR_MODEL_PATH=/models -e ASR_QUANTIZATION=int8 \
		"$IMAGE" >/dev/null
	started=1
fi
URL="http://127.0.0.1:$PORT"
until curl -sf "$URL/openapi.json" >/dev/null; do sleep 2; done

curl -sf "$URL/docs" >"$PKG/static/docs.html"
curl -sf "$URL/redoc" >"$PKG/static/redoc.html"
curl -sf "$URL/docs/oauth2-redirect" >"$PKG/static/oauth2-redirect.html"
curl -sf "$URL/openapi.json" | cmp - "$PKG/static/openapi-faster_whisper.json"

cd "$PKG" && LEGACY_REFERENCE_URL="$URL" go test -run TestCaptureReference -count=1 .
