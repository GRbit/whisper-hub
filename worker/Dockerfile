# syntax=docker/dockerfile:1
# CPU image of the whisper-hub engine worker. One engine per image:
#   docker build --build-arg ENGINE=faster_whisper -t whisper-hub-worker:faster_whisper worker
# ENGINE: faster_whisper (default, no torch) | whisperx | openai_whisper | fake
# Target "test" adds pytest and the test suite on top of the runtime image.

FROM python:3.12-slim-bookworm AS runtime

ARG ENGINE=faster_whisper
# Only the torch engines use it; CPU wheels keep the image free of CUDA libraries.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements/ requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    case "${ENGINE}" in \
      whisperx|openai_whisper) extra="--extra-index-url ${TORCH_INDEX}" ;; \
      fake|faster_whisper) extra="" ;; \
      *) echo "unknown ENGINE=${ENGINE}" >&2; exit 1 ;; \
    esac \
    && pip install ${extra} -c requirements/constraints.txt -c requirements/constraints-cpu.txt -r "requirements/${ENGINE}.txt" \
    && if [ -f "requirements/${ENGINE}-nodeps.txt" ]; then pip install --no-deps -r "requirements/${ENGINE}-nodeps.txt"; fi

COPY whisper_worker/ whisper_worker/

ENV ASR_ENGINE=${ENGINE}
EXPOSE 9100
ENTRYPOINT ["python", "-m", "whisper_worker"]

FROM runtime AS test
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements/test.txt
COPY tests/ tests/
ENTRYPOINT ["python", "-m", "pytest", "-p", "no:cacheprovider", "tests"]
CMD ["-q"]

# Last stage is what a plain `docker build` produces: the runtime image.
FROM runtime AS final
