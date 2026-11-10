#!/usr/bin/env bash
# Regenerates the legacy package's ground truth from the reference service:
#   static/openapi-*.json  upstream's schema per engine (gen_openapi.py)
#   static/*.html, assets  Swagger UI pages and bundled files
#   testdata/reference/cases/*.json  real responses for reference_test.go
#
# The reference is public upstream whisper-asr-webservice at the commit
# pinned in deploy/e2e/reference/upstream.env, built with
# `make image-reference` and run exactly as built (its own app/, from /app
# so /assets is served). whisper-hub's own extensions to upstream are
# applied on top of these captures in reference_test.go, not recorded here.
#
# Env: REF_IMAGE (default from upstream.env), PORT, MODELS_DIR (model cache).
# If a container named $NAME is already running it is reused and kept.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/../.." && pwd)
user_image=${REF_IMAGE:-}
# shellcheck source=../../../../deploy/e2e/reference/upstream.env
. "$PKG/../../deploy/e2e/reference/upstream.env"
IMAGE=${user_image:-$REF_IMAGE}
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
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
	echo "capture.sh: reference image $IMAGE not found; build it with: make image-reference" >&2
	exit 2
fi

gen() {
	docker run --rm -w /app -e PYTHONPATH=/app \
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
	docker run -d --name "$NAME" -p "127.0.0.1:$PORT:9000" \
		-v "$MODELS_DIR:/models" \
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
