ARG BASE_IMAGE=dustynv/onnxruntime:r35.4.1
FROM ${BASE_IMAGE}

WORKDIR /app

# Do not install onnxruntime from PyPI: Jetson-containers supplies the CUDA ABI
# matched build already present in this L4T R35 image.
COPY deploy/jetson/requirements.websocket.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY koko ./koko
COPY tts_model/voices_v3_turbo.json ./tts_model/voices_v3_turbo.json
COPY tts_model/more_voices.json ./tts_model/more_voices.json

EXPOSE 6942
CMD ["python3", "-m", "koko.server", "--config", "/app/config.toml"]
