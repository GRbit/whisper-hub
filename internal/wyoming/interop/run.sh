#!/usr/bin/env bash
# Interop check: the real wyoming Python package (and a replay of Home
# Assistant's STT client) against cmd/fakeserver, inside docker.
#   internal/wyoming/interop/run.sh [wyoming-version ...]   (default 1.11.0)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"; docker rm -f wyo-fake-plain wyo-fake-stream wyo-fake-fail wyo-fake-zc >/dev/null 2>&1 || true; docker network rm wyo-interop >/dev/null 2>&1 || true' EXIT

(cd "$root" && CGO_ENABLED=0 go build -o "$work/fakeserver" ./internal/wyoming/cmd/fakeserver)
docker network create wyo-interop >/dev/null
run_server() { # name, extra flags...
	local name=$1; shift
	docker run -d --rm --name "$name" --network wyo-interop -v "$work/fakeserver:/fakeserver:ro" \
		python:3.12-slim /fakeserver -debug "$@" >/dev/null
}
run_server wyo-fake-plain
run_server wyo-fake-stream -stream
run_server wyo-fake-fail -fail forbidden
run_server wyo-fake-zc -zeroconf -zeroconf-name "whisper hub test"
sleep 1

status=0
for v in "${@:-1.11.0}"; do
	img="whisper-hub-wyoming-interop:$v"
	docker build -q --build-arg "WYOMING_VERSION=$v" -t "$img" "$here" >/dev/null
	echo "== wyoming $v"
	docker run --rm --network wyo-interop "$img" || status=1
	zc_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' wyo-fake-zc)"
	docker run --rm --network wyo-interop --entrypoint python3 "$img" -I /zeroconf_check.py "$zc_ip" || status=1
done
if [ "$status" != 0 ]; then
	for n in wyo-fake-plain wyo-fake-stream wyo-fake-fail wyo-fake-zc; do echo "--- $n log"; docker logs "$n" 2>&1 | tail -40; done
fi
exit $status
