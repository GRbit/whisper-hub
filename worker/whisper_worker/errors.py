"""Client-visible errors of the Worker API."""


class WorkerError(Exception):
    """WorkerError is an error whose code and message are safe to send to the client.

    Never put file paths, library messages or tracebacks into message: those
    go to the log only.
    """

    status = 500
    code = "internal"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def body(self) -> dict:
        return {"code": self.code, "message": self.message}


class BadRequest(WorkerError):
    """BadRequest means the request itself is invalid."""

    status = 400
    code = "bad_request"


class ModelNotFound(WorkerError):
    """ModelNotFound means the worker does not serve the requested model."""

    status = 404
    code = "model_not_found"


class Unavailable(WorkerError):
    """Unavailable means the job could not run now, for example the model failed to load."""

    status = 503
    code = "unavailable"


class NotPausable(WorkerError):
    """NotPausable refuses a pause request the worker could not honor."""

    status = 409
    code = "not_pausable"


class Internal(WorkerError):
    """Internal means the engine failed while running the job."""

    status = 500
    code = "internal"


class JobCancelled(Exception):
    """JobCancelled is raised inside an engine when the client went away."""
