#!/usr/bin/env bash
# Prints the image tags for one build, one per line.
#   image-tags.sh <repository> <variant> <default:true|false>
# repository: e.g. ghcr.io/grbit/whisper-hub; variant: e.g. faster_whisper,
# whisperx-gpu, or "" for the gateway; default: the variant also gets the
# plain tags (latest, edge, <version>).
# From GITHUB_REF and GITHUB_SHA:
#   tag vX.Y.Z    X.Y.Z-<variant>, <variant>;  default variant also X.Y.Z, latest
#   branch master edge-<variant>, sha-<7>-<variant>;  default variant also edge, sha-<7>
#   anything else (pull requests): pr-<variant>, never pushed
set -euo pipefail
repo=$1 variant=$2 default=$3
ref=${GITHUB_REF:-} sha=${GITHUB_SHA:-0000000}
short=${sha:0:7}
sfx=${variant:+-$variant}
tags=()
case "$ref" in
refs/tags/v*)
	ver=${ref#refs/tags/v}
	tags+=("$ver$sfx")
	[ -n "$variant" ] && tags+=("$variant")
	[ "$default" = true ] && tags+=("$ver" latest)
	;;
refs/heads/master)
	tags+=("edge$sfx" "sha-$short$sfx")
	[ "$default" = true ] && tags+=(edge "sha-$short")
	;;
*)
	tags+=("pr$sfx")
	;;
esac
# Drop duplicates (the gateway has no variant, so its plain and suffixed tags coincide).
printf '%s\n' "${tags[@]}" | awk '!seen[$0]++' | sed "s|^|$repo:|"
