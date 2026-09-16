# Container Deployment

This deployment runs the validated local inference API in a CPU-only container.
Model artifacts, taxonomy data, and Ego4D frames remain on the host and are
mounted read-only. They are not copied into the image.

## Start on Windows

Docker Desktop must be running with Linux containers enabled. From the
repository root in PowerShell:

```powershell
Copy-Item deploy/.env.example deploy/.env
docker compose --env-file deploy/.env -f deploy/compose.yaml build
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
```

Check liveness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The request path must use the container's frame mount, not the Windows host
path. For example:

```powershell
$body = @{
    target_frame = "/data/frames/002e11bc-deef-45f7-9af8-59421a606d69/0008939.jpg"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body $body
```

Inspect logs or stop the service:

```powershell
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f
docker compose --env-file deploy/.env -f deploy/compose.yaml down
```

The published port remains bound to `127.0.0.1`; the API is not exposed to the
network. Remote exposure requires a separately configured authenticated TLS
reverse proxy.
