# Local Inference Interfaces

The production inference stack supports a one-shot JSON CLI and a local HTTP
API. Both interfaces return the same prediction schema.

## Requirements

Run commands from the repository root with the project virtual environment.
The configured Track A and Track B checkpoints, taxonomy, TTC statistics, and
feature contract must exist. Token extraction also requires an explicit
ResNet18 `IMAGENET1K_V1` weight artifact.

## Command line

```powershell
.\local_extraction\.venv\Scripts\python.exe -m local_extraction.inference.cli `
  D:\path\to\frames\0000123.jpg `
  --tokenizer-weights D:\path\to\resnet18-f37072fd.pth `
  --output prediction.json
```

## Local HTTP API

Start the server bound to the local machine:

```powershell
.\local_extraction\.venv\Scripts\python.exe -m local_extraction.inference.api `
  --host 127.0.0.1 `
  --port 8000 `
  --tokenizer-weights D:\path\to\resnet18-f37072fd.pth
```

The health endpoint does not load model weights:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Submit a target frame already available on the server machine:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType application/json `
  -Body '{"target_frame":"D:\\path\\to\\frames\\0000123.jpg"}'
```

The predictor is initialized on the first prediction and reused afterward.
Requests are serialized because the model stack is stateful and intended for
single-device local inference. The default `127.0.0.1` binding does not expose
the service to other machines.
