#!/usr/bin/env bash
# Real clients against the real gateway image, with a fake engine worker:
#   - the openai Python SDK: internal/openaiapi/cmd/fakeserver/sdk_check.py
#   - the wyoming package and a Home Assistant STT replay:
#     internal/wyoming/interop/client.py and zeroconf_check.py
# Those checks run unmodified; only their targets change (the gateway
# containers take the host names the scripts expect).
#   GATEWAY_IMAGE=whisper-hub-gateway:latest deploy/e2e/clients.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
image="${GATEWAY_IMAGE:-whisper-hub-gateway:latest}"
sdk_image="${E2E_IMAGE:-whisper-hub-e2e:latest}"
wyoming_version="${WYOMING_VERSION:-1.11.0}"
net=whub-e2e
names=(whub-fw-openai whub-fw-heard whub-gw-openai wyo-fake-plain wyo-fake-stream wyo-fake-fail wyo-fake-zc)
work="$(mktemp -d)"
cleanup() {
	rm -rf "$work"
	docker rm -f "${names[@]}" >/dev/null 2>&1 || true
	docker network rm "$net" >/dev/null 2>&1 || true
}
trap cleanup EXIT

(cd "$root" && CGO_ENABLED=0 go build -o "$work/fakeworker" ./deploy/e2e/fakeworker)
chmod 755 "$work"
docker network create "$net" >/dev/null

fakeworker() { # name script
	docker run -d --name "$1" --network "$net" -v "$work/fakeworker:/fakeworker:ro" \
		--entrypoint /fakeworker "$image" -script "$2" >/dev/null
}
gateway() { # name worker-name env...
	local name=$1 worker=$2; shift 2
	local envs=(-e "WHISPER_HUB_HUB_WORKERS=[{url: \"http://$worker:9100\"}]" -e WHISPER_HUB_LOG_LEVEL=debug)
	for e in "$@"; do envs+=(-e "$e"); done
	docker run -d --name "$name" --network "$net" -v "$work:/work:ro" "${envs[@]}" "$image" >/dev/null
}
wait_healthy() { # name
	for _ in $(seq 100); do
		if docker exec "$1" whisper-hub healthcheck >/dev/null 2>&1; then return 0; fi
		sleep 0.2
	done
	echo "$1 never became healthy"; docker logs "$1" 2>&1 | tail -30; return 1
}

# --- OpenAI SDK ---
# The script authenticates with the fixed key "sk-test"; model "forbidden"
# is served by the worker but not granted, "missing" is unknown and
# unknown names are an error here, as the script expects.
key_hash="sha256:$(printf %s sk-test | sha256sum | cut -d' ' -f1)"
cat >"$work/users.yaml" <<EOF
users:
  - name: sdk
    api_keys: ["$key_hash"]
    models: [base, busy, boom, midfail]
EOF
chmod 644 "$work/users.yaml"
# 8 kHz keeps 3.5 s under the 100000-byte upload limit the script tests.
ffmpeg -v error -f lavfi -i "sine=frequency=440:duration=3.5" -ar 8000 "$work/a.wav"
printf 'corrupt, not audio' >"$work/corrupt.wav"
chmod 644 "$work"/*.wav
cp "$root/internal/openaiapi/cmd/fakeserver/sdk_check.py" "$work/"

fakeworker whub-fw-openai openai
fakeworker whub-fw-heard heard
gateway whub-gw-openai whub-fw-openai WHISPER_HUB_AUTH_USERS_FILE=/work/users.yaml \
	WHISPER_HUB_HUB_UNKNOWN_MODEL=error WHISPER_HUB_OPENAI_MAX_UPLOAD_BYTES=100000 WHISPER_HUB_WYOMING_ENABLED=false
gateway wyo-fake-plain whub-fw-heard
gateway wyo-fake-stream whub-fw-heard WHISPER_HUB_WYOMING_STREAM_TRANSCRIPTS=true
gateway wyo-fake-fail whub-fw-heard WHISPER_HUB_AUTH_WYOMING_ALLOW_CIDRS=203.0.113.0/24
gateway wyo-fake-zc whub-fw-heard WHISPER_HUB_WYOMING_ZEROCONF_ENABLED=true "WHISPER_HUB_WYOMING_ZEROCONF_NAME=whisper hub test"
for n in whub-gw-openai wyo-fake-plain wyo-fake-stream wyo-fake-fail wyo-fake-zc; do wait_healthy "$n"; done

status=0
echo "== openai SDK"
docker run --rm --network "$net" -v "$work:/work:ro" "$sdk_image" /work/sdk_check.py \
	--real-gateway http://whub-gw-openai:9000/v1 /work/a.wav || status=1

echo "== wyoming $wyoming_version"
interop="whisper-hub-wyoming-interop:$wyoming_version"
docker build -q --build-arg "WYOMING_VERSION=$wyoming_version" -t "$interop" "$root/internal/wyoming/interop" >/dev/null
docker run --rm --network "$net" "$interop" || status=1
zc_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' wyo-fake-zc)"
docker run --rm --network "$net" --entrypoint python3 "$interop" -I /zeroconf_check.py "$zc_ip" || status=1

if [ "$status" != 0 ]; then
	for n in "${names[@]}"; do echo "--- $n log"; docker logs "$n" 2>&1 | tail -30; done
fi
exit $status
