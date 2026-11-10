#!/usr/bin/env bash
# whisper-hub:all runs the same worker code as the single-engine images, so
# it behaves the same exactly when every package of each single-engine
# image is installed in it at the same version. This checks that, in
# seconds, instead of rerunning the whole-stack fidelity comparison per
# ASR_ENGINE value.
#   deploy/e2e/packages.sh            ALL_IMAGE=whisper-hub:all, ENGINES="faster_whisper openai_whisper whisperx"
# Exit status 1 lists every package missing from or differing in :all.
set -euo pipefail
all_image="${ALL_IMAGE:-whisper-hub:all}"
engines="${ENGINES:-faster_whisper openai_whisper whisperx}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

freeze() { docker run --rm --entrypoint python "$1" -m pip freeze --all 2>/dev/null >"$2"; }
freeze "$all_image" "$work/all.txt"
status=0
for e in $engines; do
	freeze "whisper-hub:$e" "$work/$e.txt"
	python3 - "$work/$e.txt" "$work/all.txt" "whisper-hub:$e" "$all_image" <<'PY' || status=1
import re, sys

def load(path):
    pkgs = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, ver = line.partition("==")
        if not sep:  # "name @ url" and the like: compare the whole spec
            name, ver = line.split(" ", 1)[0], line
        pkgs[re.sub(r"[-_.]+", "-", name).lower()] = ver
    return pkgs

single, full = load(sys.argv[1]), load(sys.argv[2])
bad = [(n, v, full.get(n)) for n, v in sorted(single.items()) if full.get(n) != v]
for n, want, got in bad:
    print(f"FAIL {sys.argv[3]}: {n} {want} -> {sys.argv[4]}: {got or 'missing'}")
print(("ok   " if not bad else "FAIL ") + f"{sys.argv[3]}: {len(single)} packages, {len(bad)} missing or different in {sys.argv[4]} ({len(full)} packages)")
sys.exit(1 if bad else 0)
PY
done
exit $status
