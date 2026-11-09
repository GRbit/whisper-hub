# Python side of the Wyoming interop check; see run.sh.
FROM python:3.12-slim
ARG WYOMING_VERSION=1.11.0
RUN pip install --no-cache-dir "wyoming==${WYOMING_VERSION}" "zeroconf==0.154.0"
COPY client.py zeroconf_check.py /
ENTRYPOINT ["python3", "-I", "/client.py"]
