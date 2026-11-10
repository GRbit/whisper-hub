#!/usr/bin/env bash
# End-to-end runs against real containers (see stack_check.py):
#   deploy/e2e/stack.sh fake   gateway image + Python fake worker image: protocols,
#                              streaming, cancellation, priority (usage log), auth on and off
#   deploy/e2e/stack.sh real   all-in-one image, model tiny on CPU, real audio, plus the
#                              sidecar compatibility mode (SIDECAR_*). ENGINE picks the
#                              image (whisper-hub:$ENGINE), ASR_ENGINE is passed to it if set.
# Images: GATEWAY_IMAGE, WORKER_IMAGE, AIO_IMAGE, E2E_IMAGE (built by make).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
mode="${1:?usage: stack.sh fake|real}"
gateway_image="${GATEWAY_IMAGE:-whisper-hub-gateway:latest}"
worker_image="${WORKER_IMAGE:-whisper-hub-worker:fake}"
aio_image="${AIO_IMAGE:-whisper-hub:${ENGINE:-faster_whisper}}"
aio_env=(-e ASR_MODEL=tiny)
[ -n "${ASR_ENGINE:-}" ] && aio_env+=(-e "ASR_ENGINE=$ASR_ENGINE")
e2e_image="${E2E_IMAGE:-whisper-hub-e2e:latest}"
cache_volume="${CACHE_VOLUME:-whub-e2e-cache}"
voice="${VOICE_FILES:-/path/to/voice-test-files}"
net=whub-stack
names=(whub-s-worker whub-s-gw whub-s-gw-auth whub-s-aio whub-s-aio-proxy)
work="$(mktemp -d)"
cleanup() {
	docker rm -f "${names[@]}" >/dev/null 2>&1 || true
	docker network rm "$net" >/dev/null 2>&1 || true
	rm -rf "$work"
}
trap cleanup EXIT
docker network create "$net" >/dev/null

mkdir -p "$work/audio" "$work/data"
chmod 777 "$work/data"
if [ -f "$voice/english-front-center.wav" ]; then
	cp "$voice/english-front-center.wav" "$voice/german-20s.wav" "$work/audio/"
elif [ "$mode" = fake ]; then
	# CI has no recordings; the fake engine ignores the content anyway.
	voice="$work/audio"
	ffmpeg -v error -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=1.428" -ac 2 "$voice/english-front-center.wav"
	ffmpeg -v error -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=20" -ac 2 "$voice/german-20s.wav"
else
	echo "real mode needs the recordings in VOICE_FILES ($voice)" >&2
	exit 2
fi
ffmpeg -v error -i "$voice/english-front-center.wav" -f s16le -ac 1 -ar 16000 "$work/audio/english-front-center.pcm"
# ffmpeg's default MP4 layout puts the moov atom last, as Android MediaMuxer does.
ffmpeg -v error -i "$voice/german-20s.wav" -c:a aac -b:a 96k "$work/audio/german-trailing-moov.m4a"
chmod -R a+rX "$work"

wait_healthy() { # container port
	for _ in $(seq 300); do
		if docker exec "$1" whisper-hub healthcheck --port "$2" >/dev/null 2>&1; then return 0; fi
		sleep 1
	done
	echo "$1 never became healthy"; docker logs "$1" 2>&1 | tail -30; return 1
}
check() { # container args...
	docker run --rm --network "$net" -v "$work/audio:/audio:ro" -v "$root/internal/wyoming/interop:/interop:ro" \
		-v "$here/stack_check.py:/stack_check.py:ro" "$e2e_image" /stack_check.py "$@"
}

# Users: alice with a password (legacy Basic) and an API key (OpenAI);
# homeassistant is the Wyoming identity.
key="wh_e2e_$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
key_hash="sha256:$(printf %s "$key" | sha256sum | cut -d' ' -f1)"
pw_hash="$(printf 'e2e-secret\n' | docker run --rm -i --entrypoint whisper-hub "$gateway_image" hash)"
cat >"$work/users.yaml" <<YAML
users:
  - name: alice
    password_hash: "$pw_hash"
    api_keys: ["$key_hash"]
  - name: homeassistant
YAML
chmod 644 "$work/users.yaml"

status=0
if [ "$mode" = fake ]; then
	docker run -d --name whub-s-worker --network "$net" "$worker_image" >/dev/null
	gw_env=(-e 'WHISPER_HUB_HUB_WORKERS=[{url: "http://whub-s-worker:9100"}]' -e LOG_LEVEL=DEBUG)
	docker run -d --name whub-s-gw --network "$net" -v "$work/data:/data" "${gw_env[@]}" \
		-e WHISPER_HUB_AUTH_USAGE_LOG_FILE=/data/requests.jsonl "$gateway_image" >/dev/null
	docker run -d --name whub-s-gw-auth --network "$net" -v "$work:/work:ro" "${gw_env[@]}" \
		-e WHISPER_HUB_AUTH_USERS_FILE=/work/users.yaml -e WHISPER_HUB_AUTH_WYOMING_USER=homeassistant "$gateway_image" >/dev/null
	wait_healthy whub-s-gw 9000
	wait_healthy whub-s-gw-auth 9000

	echo "== fake worker, auth off"
	check --mode fake --http http://whub-s-gw:9000 --wyoming whub-s-gw:10300 || status=1
	echo "== usage log: priority order"
	# The log is private to the gateway user (0600), so read it through the container.
	docker exec whub-s-gw cat /data/requests.jsonl >"$work/requests.jsonl"
	python3 - "$work/requests.jsonl" <<'PY' || status=1
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1])]
# The priority scenario is the last four jobs; the log is written at completion.
last = lines[-4:]
order = [l["endpoint"] for l in last]
w = next(l for l in last if l["protocol"] == "wyoming")
long_, q1, q2 = [l for l in last if l["protocol"] == "legacy"]
ok = (order == ["/asr", "transcribe", "/asr", "/asr"] and w["queued_ms"] < long_["elapsed_ms"]
      and q1["queued_ms"] > w["queued_ms"] and q2["queued_ms"] > w["queued_ms"])
print(("ok   " if ok else "FAIL ") + "usage log: wyoming queued %d ms behind the running job only; queued jobs waited %d and %d ms"
      % (w["queued_ms"], q1["queued_ms"], q2["queued_ms"]))
sys.exit(0 if ok else 1)
PY
	echo "== fake worker, auth on (Basic for legacy, API key for OpenAI, Wyoming as homeassistant)"
	check --mode fake --http http://whub-s-gw-auth:9000 --wyoming whub-s-gw-auth:10300 \
		--basic alice:e2e-secret --key "$key" --only legacy,sdk,wyoming || status=1
	code=$(docker run --rm --network "$net" --entrypoint python3 "$e2e_image" -c \
		"import urllib.request,urllib.error
try: urllib.request.urlopen(urllib.request.Request('http://whub-s-gw-auth:9000/asr', data=b'x'))
except urllib.error.HTTPError as e: print(e.code)")
	[ "$code" = 401 ] && echo "ok   legacy without credentials -> 401" || { echo "FAIL legacy without credentials -> $code"; status=1; }
else
	docker run -d --name whub-s-aio --network "$net" -v "$cache_volume:/root/.cache" "${aio_env[@]}" "$aio_image" >/dev/null
	wait_healthy whub-s-aio 9000
	# The engine actually running, from the gateway's "worker up" line.
	running="$(docker logs whub-s-aio 2>&1 | grep 'msg="worker up"' | grep -o 'engine=[a-z_]*' | head -1)"
	echo "== all-in-one $aio_image ${ASR_ENGINE:+ASR_ENGINE=$ASR_ENGINE }-> $running, model tiny"
	check --mode real --http http://whub-s-aio:9000 --wyoming whub-s-aio:10300 || status=1
	docker rm -f whub-s-aio >/dev/null

	# The old sidecar deployment, unchanged: bcrypt users file at
	# SIDECAR_CONFIG, authenticated 9001, open 9000, usage in SIDECAR_USAGE_DIR.
	printf 'users:\n  - name: alice\n    password_hash: "%s"\n' "$pw_hash" >"$work/auth.yaml"
	chmod 644 "$work/auth.yaml"
	docker run -d --name whub-s-aio-proxy --network "$net" -v "$cache_volume:/root/.cache" -v "$work/auth.yaml:/config/auth.yaml:ro" \
		-v "$work/data:/data/sidecar" "${aio_env[@]}" -e SIDECAR_ENABLED=true -e SIDECAR_SNAPSHOT_INTERVAL=1s "$aio_image" >/dev/null
	wait_healthy whub-s-aio-proxy 9001
	echo "== sidecar compatibility: Basic auth on 9001"
	check --mode real --http http://whub-s-aio-proxy:9001 --wyoming whub-s-aio-proxy:10300 --basic alice:e2e-secret --only legacy || status=1
	echo "== sidecar compatibility: open port 9000"
	check --mode real --http http://whub-s-aio-proxy:9000 --wyoming whub-s-aio-proxy:10300 --only legacy || status=1
	code=$(docker run --rm --network "$net" --entrypoint python3 "$e2e_image" -c \
		"import urllib.request,urllib.error
try: urllib.request.urlopen(urllib.request.Request('http://whub-s-aio-proxy:9001/asr', data=b'x'))
except urllib.error.HTTPError as e: print(e.code)")
	[ "$code" = 401 ] && echo "ok   9001 without credentials -> 401" || { echo "FAIL 9001 without credentials -> $code"; status=1; }
	sleep 2
	if python3 -c "import json,sys; d=json.load(open('$work/data/usage.json')); sys.exit(0 if d['alice']['ok_requests']>0 else 1)" 2>/dev/null \
		|| python3 -c "import json,sys; d=json.load(open('$work/data/usage.json')); sys.exit(0 if d['users']['alice']['ok_requests']>0 else 1)"; then
		echo "ok   usage.json in SIDECAR_USAGE_DIR counts alice"
	else
		echo "FAIL usage.json"; ls -la "$work/data"; status=1
	fi
fi
if [ "$status" != 0 ]; then
	for n in "${names[@]}"; do docker logs "$n" >/dev/null 2>&1 && { echo "--- $n"; docker logs "$n" 2>&1 | grep -v DEBUG | tail -25; }; done
fi
exit $status
