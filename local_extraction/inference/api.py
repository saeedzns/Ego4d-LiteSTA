"""Dependency-free HTTP API for local short-term anticipation inference."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence

from .cli import result_to_dict
from .config import InferenceConfig


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_REQUEST_BYTES = 64 * 1024


class InferenceAPIError(RuntimeError):
    """Raised when an API prediction request cannot be completed."""


class InferenceService:
    """Lazily construct and serialize access to the inference pipeline."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        predictor_factory: Callable[[InferenceConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self._predictor_factory = predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            if self._predictor_factory is None:
                from .pipeline import ShortTermAnticipationPredictor

                self._predictor_factory = ShortTermAnticipationPredictor
            self._predictor = self._predictor_factory(self.config)
        return self._predictor

    def predict(self, target_frame: str | Path) -> dict[str, Any]:
        """Run one request and return the stable JSON-ready response."""
        with self._lock:
            try:
                result = self._get_predictor().predict(Path(target_frame))
            except Exception as exc:
                raise InferenceAPIError(f"Local inference failed: {exc}") from exc
        return result_to_dict(result)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def _handler_factory(service: InferenceService) -> type[BaseHTTPRequestHandler]:
    class InferenceRequestHandler(BaseHTTPRequestHandler):
        server_version = "Ego4DLiteSTA/1.0"

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "Content-Type must be application/json"},
                )
                return

            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                content_length = -1
            if content_length < 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Content-Length must be between 0 and {MAX_REQUEST_BYTES}"},
                )
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Malformed JSON body"})
                return

            target_frame = payload.get("target_frame") if isinstance(payload, dict) else None
            if not isinstance(target_frame, str) or not target_frame.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "target_frame must be a non-empty string"},
                )
                return

            try:
                result = service.predict(target_frame)
            except InferenceAPIError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return InferenceRequestHandler


def create_server(
    service: InferenceService,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Create an HTTP server without starting its request loop."""
    return ThreadingHTTPServer((host, port), _handler_factory(service))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local Ego4D inference API.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer-weights", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = InferenceConfig(
        device=args.device,
        tokenizer_weights_path=args.tokenizer_weights,
    )
    server = create_server(InferenceService(config), host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InferenceAPIError",
    "InferenceService",
    "build_parser",
    "create_server",
    "main",
]
