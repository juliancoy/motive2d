FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install "numpy<2" ultralytics opencv-python

WORKDIR /work
COPY run_yolo11.sh /work/run_yolo11.sh
RUN chmod +x /work/run_yolo11.sh

ENTRYPOINT ["bash", "/work/run_yolo11.sh"]
