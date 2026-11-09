# Client side of deploy/e2e: the openai SDK and the wyoming package, at the
# versions the frontend owners checked against.
FROM python:3.12-slim
RUN pip install --no-cache-dir "openai==3.48.0" "wyoming==1.11.0" "zeroconf==0.154.0"
ENTRYPOINT ["python3", "-I"]
