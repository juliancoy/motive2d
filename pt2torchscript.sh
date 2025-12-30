# run this in torch:latest
pip install -U ultralytics
apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
python -c 'from ultralytics import YOLO; YOLO("models/yolo11n-pose.pt").export(format="torchscript")'
pip install torchinfo
python - <<'PY'
    import torch
    from torchinfo import summary  # install via pip if missing
    model = torch.jit.load("models/yolo11n-pose.torchscript")
    summary(model, input_size=(1, 3, 640, 640))
PY