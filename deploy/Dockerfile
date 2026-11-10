# syntax=docker/dockerfile:1
# All-in-one image: gateway + one engine worker, a drop-in replacement for
# the upstream whisper-asr-webservice image (port 9000, same environment
# variables and flags, model cache in /root/.cache/whisper), plus Wyoming
# on 10300.
#   docker build -f deploy/Dockerfile -t whisper-hub:faster_whisper .
#   docker build -f deploy/Dockerfile --build-arg ENGINE=openai_whisper -t whisper-hub:openai_whisper .
# ENGINE: faster_whisper (default, no torch) | whisperx | openai_whisper | all | fake
# "all" installs the three real engines; ASR_ENGINE picks one at startup and
# defaults to openai_whisper there, as upstream.
#
# The gateway is PID 1 and starts the worker as its child (worker.command):
# signals reach the process that knows the shutdown order, and the
# container exits when either process dies.
ARG ENGINE=faster_whisper

FROM onerahmet/ffmpeg:n7.1 AS ffmpeg

FROM golang:1.25-alpine AS gateway
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY cmd/ cmd/
COPY internal/ internal/
ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/whisper-hub ./cmd/whisper-hub

# Same base and install steps as worker/Dockerfile, so the heavy pip layer
# is shared with the worker image.
FROM python:3.12-slim-bookworm
ARG ENGINE
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY worker/requirements/ requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    case "${ENGINE}" in \
      whisperx|openai_whisper|all) extra="--extra-index-url ${TORCH_INDEX}" ;; \
      fake|faster_whisper) extra="" ;; \
      *) echo "unknown ENGINE=${ENGINE}" >&2; exit 1 ;; \
    esac \
    && pip install ${extra} -c requirements/constraints.txt -r "requirements/${ENGINE}.txt" \
    && if [ -f "requirements/${ENGINE}-nodeps.txt" ]; then pip install --no-deps -r "requirements/${ENGINE}-nodeps.txt"; fi
COPY worker/whisper_worker/ whisper_worker/

COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=gateway /out/whisper-hub /usr/local/bin/whisper-hub
RUN ln -s whisper-hub /usr/local/bin/whisper-asr-webservice

# ASR_ENGINE is deliberately not set: unset means "this image's engine"
# (upstream defaults to openai_whisper), and a different value fails at
# startup naming the image that has it.
# The worker port is internal and unusual so an upstream-style
# "--port 9100" cannot collide with it. The model loads at startup, as
# upstream does.
ENV WHISPER_HUB_WORKER_COMMAND="[python, -m, whisper_worker]" \
    WHISPER_HUB_WORKER_ENGINE=${ENGINE} \
    WHISPER_HUB_WORKER_IMAGE_TAG="whisper-hub:%s" \
    WHISPER_HUB_HUB_WORKERS="[{url: \"http://127.0.0.1:19100\"}]" \
    WORKER_HOST=127.0.0.1 \
    WORKER_PORT=19100 \
    WORKER_PRELOAD=true \
    WHISPER_HUB_HUB_RETRY_MAX=2s

# Root, like upstream: existing cache volumes on /root/.cache are root-owned.
EXPOSE 9000 10300
HEALTHCHECK --interval=30s --start-period=10m CMD ["whisper-hub", "healthcheck"]
ENTRYPOINT ["whisper-hub"]
