GO      ?= go
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)
# Engine of the all-in-one and worker images: faster_whisper, whisperx,
# openai_whisper, or all (all-in-one only). ASR_ENGINE, when set, is passed
# to the all-in-one container in the e2e runs (meaningful for ENGINE=all).
ENGINE  ?= faster_whisper
ASR_ENGINE ?=
export ENGINE ASR_ENGINE

.PHONY: build fmt vet test test-short check \
	images image image-gpu image-gateway image-worker image-worker-fake image-e2e \
	e2e e2e-clients e2e-fake e2e-real e2e-fidelity e2e-packages e2e-engines clean

build:
	$(GO) build -ldflags '$(LDFLAGS)' -o bin/whisper-hub ./cmd/whisper-hub

fmt:
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

vet:
	$(GO) vet ./...

test:
	$(GO) test -race ./...

# Skips the slow bcrypt tests of internal/auth.
test-short:
	$(GO) test -race -short ./...

check: fmt vet test

# --- images (no host toolchain needed: everything builds in docker) ---

images: image image-gateway

# All-in-one: gateway + worker, drop-in for the upstream image.
image:
	docker build -f deploy/Dockerfile --build-arg ENGINE=$(ENGINE) --build-arg VERSION=$(VERSION) -t whisper-hub:$(ENGINE) .

# CUDA all-in-one (whisper-hub:<engine>-gpu); CI builds these.
image-gpu:
	docker build -f deploy/Dockerfile.gpu --build-arg ENGINE=$(ENGINE) --build-arg VERSION=$(VERSION) -t whisper-hub:$(ENGINE)-gpu .

image-gateway:
	docker build -f deploy/Dockerfile.gateway --build-arg VERSION=$(VERSION) -t whisper-hub-gateway:latest .

image-worker:
	docker build --build-arg ENGINE=$(ENGINE) -t whisper-hub-worker:$(ENGINE) worker

image-worker-fake:
	docker build --build-arg ENGINE=fake -t whisper-hub-worker:fake worker

image-e2e:
	docker build -t whisper-hub-e2e:latest deploy/e2e

# --- end-to-end checks against real containers (see deploy/e2e) ---

e2e: e2e-clients e2e-fake e2e-real e2e-fidelity

# Owners' real-client scripts (openai SDK, wyoming, HA replay, zeroconf)
# against the gateway image with a canned Go fake worker.
e2e-clients: image-gateway image-e2e
	deploy/e2e/clients.sh

# Gateway image + Python fake worker: protocols, streaming, cancellation,
# priority, auth on and off.
e2e-fake: image-gateway image-worker-fake image-e2e
	deploy/e2e/stack.sh fake

# All-in-one whisper-hub:$(ENGINE) with model tiny on real audio, and the
# sidecar compatibility mode.
e2e-real: image image-e2e
	deploy/e2e/stack.sh real

# Byte-for-byte legacy comparison with the reference image whisper-asr-webservice:ec32cd5
# running the same engine. openai_whisper is not deterministic upstream:
# REF_TWICE=1 labels the cases where the reference disagrees with itself.
e2e-fidelity: image
	deploy/e2e/fidelity.sh

# Same package versions in whisper-hub:all as in each single-engine image
# (same worker code too), so :all needs no fidelity run of its own.
e2e-packages:
	deploy/e2e/packages.sh

# e2e-real and e2e-fidelity for every single-engine image; for
# whisper-hub:all the real-audio run per ASR_ENGINE (and unset) plus
# e2e-packages. Fidelity can still target :all by hand:
#   make e2e-fidelity ENGINE=all ASR_ENGINE=whisperx
e2e-engines: image-e2e
	for e in faster_whisper openai_whisper whisperx all; do $(MAKE) image ENGINE=$$e || exit 1; done
	for e in faster_whisper openai_whisper whisperx; do \
		ENGINE=$$e ASR_ENGINE= deploy/e2e/stack.sh real && ENGINE=$$e ASR_ENGINE= REF_TWICE=$$([ $$e = openai_whisper ] && echo 1) deploy/e2e/fidelity.sh || exit 1; done
	for a in "" faster_whisper openai_whisper whisperx; do ENGINE=all ASR_ENGINE=$$a deploy/e2e/stack.sh real || exit 1; done
	deploy/e2e/packages.sh

clean:
	rm -rf bin
