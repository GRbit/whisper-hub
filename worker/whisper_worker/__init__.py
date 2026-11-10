"""whisper_worker is the Python engine worker of whisper-hub.

It runs one speech recognition engine behind the internal Worker HTTP API
described in docs/worker-api.md. The gateway owns every client-facing protocol.
"""

__version__ = "0.1.0"
