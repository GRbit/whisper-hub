# Prints upstream's /openapi.json for the engine configured through the
# ASR_ENGINE and HF_TOKEN environment variables, without loading a model:
# the schema only depends on the FastAPI route declarations.
# Run inside the reference image, see capture.sh.
import contextlib
import sys

# app.config prints a warning for whisperx without HF_TOKEN; keep stdout clean.
with contextlib.redirect_stdout(sys.stderr):
    import app.factory.asr_model_factory as factory


class _NoModel:
    def load_model(self):
        pass


factory.ASRModelFactory.create_asr_model = staticmethod(lambda: _NoModel())

with contextlib.redirect_stdout(sys.stderr):
    from fastapi.responses import JSONResponse

    from app.webservice import app

# Render exactly as FastAPI's /openapi.json route does.
sys.stdout.buffer.write(JSONResponse(app.openapi()).body)
