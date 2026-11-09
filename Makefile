GO        ?= go
VERSION   ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS   := -s -w -X main.version=$(VERSION)
GATEWAY_IMAGE ?= whisper-hub-gateway:latest

.PHONY: build build-static fmt vet test test-short check image-gateway e2e-docker clean

build:
	$(GO) build -ldflags '$(LDFLAGS)' -o bin/whisper-hub ./cmd/whisper-hub

# Static binary for the container images.
build-static:
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o bin/whisper-hub ./cmd/whisper-hub

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

image-gateway: build-static
	docker build -f deploy/Dockerfile.gateway -t $(GATEWAY_IMAGE) bin

# Real clients (openai SDK, wyoming package, Home Assistant replay) against
# the gateway image and a fake worker, in docker.
e2e-docker: image-gateway
	GATEWAY_IMAGE=$(GATEWAY_IMAGE) deploy/e2e/run.sh

clean:
	rm -rf bin
