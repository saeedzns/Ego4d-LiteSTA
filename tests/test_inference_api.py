import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from local_extraction.inference import InferenceConfig
from local_extraction.inference.api import InferenceAPIError, InferenceService, create_server


def _result(tmp_path: Path) -> SimpleNamespace:
    frame = tmp_path / "uid" / "0000001.jpg"
    candidate = SimpleNamespace(box_xyxy=(1.0, 2.0, 3.0, 4.0), confidence=0.75, class_id=9)
    prediction = SimpleNamespace(
        candidate=candidate,
        candidate_index=0,
        score=0.8,
        detector_confidence=0.75,
        next_active_probability=0.8,
        predicted_class=1,
        noun_id=10,
        noun_local_index=2,
        noun_probability=0.6,
        verb_id=20,
        verb_local_index=1,
        verb_probability=0.7,
        ttc_seconds=1.25,
        ttc_raw=2.0,
        ttc_bin=2,
    )
    return SimpleNamespace(
        resolved_input=SimpleNamespace(
            target_frame=frame,
            uid="uid",
            frame_directory=frame.parent,
            context_frames=(frame,),
        ),
        image_size=(480, 640),
        candidates=(candidate,),
        predictions=(prediction,),
    )


class FakePredictor:
    def __init__(self, config: InferenceConfig, result: object) -> None:
        self.config = config
        self.result = result
        self.calls: list[Path] = []

    def predict(self, target_frame: Path) -> object:
        self.calls.append(target_frame)
        return self.result


def _request(url: str, *, method: str = "GET", payload: object | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        response = urlopen(request, timeout=2)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read())


@pytest.fixture
def api_server(tmp_path: Path):
    result = _result(tmp_path)
    instances: list[FakePredictor] = []

    def factory(config: InferenceConfig) -> FakePredictor:
        predictor = FakePredictor(config, result)
        instances.append(predictor)
        return predictor

    service = InferenceService(InferenceConfig(), predictor_factory=factory)
    server = create_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", instances
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_does_not_initialize_predictor(api_server) -> None:
    base_url, instances = api_server

    status, payload = _request(f"{base_url}/health")

    assert status == 200
    assert payload == {"status": "ok"}
    assert instances == []


def test_predict_returns_cli_compatible_schema(api_server) -> None:
    base_url, instances = api_server

    status, payload = _request(
        f"{base_url}/predict",
        method="POST",
        payload={"target_frame": "frames/0000001.jpg"},
    )

    assert status == 200
    assert payload["prediction_count"] == 1
    assert payload["predictions"][0]["noun_id"] == 10
    assert instances[0].calls == [Path("frames/0000001.jpg")]


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"target_frame": ""}, {"target_frame": 123}],
)
def test_predict_rejects_invalid_payload(api_server, payload: object) -> None:
    base_url, instances = api_server
    request_payload = payload if payload is not None else None
    if request_payload is None:
        request = Request(f"{base_url}/predict", data=b"null", method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.status
    else:
        status, _ = _request(f"{base_url}/predict", method="POST", payload=request_payload)

    assert status == 400
    assert instances == []


def test_predict_rejects_wrong_content_type(api_server) -> None:
    base_url, _ = api_server
    request = Request(f"{base_url}/predict", data=b"{}", method="POST")

    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=2)

    assert error.value.code == 415


def test_unknown_route_returns_not_found(api_server) -> None:
    base_url, _ = api_server

    status, payload = _request(f"{base_url}/missing")

    assert status == 404
    assert payload == {"error": "Not found"}


def test_service_reuses_predictor(tmp_path: Path) -> None:
    result = _result(tmp_path)
    instances: list[FakePredictor] = []

    def factory(config: InferenceConfig) -> FakePredictor:
        predictor = FakePredictor(config, result)
        instances.append(predictor)
        return predictor

    service = InferenceService(InferenceConfig(), predictor_factory=factory)
    service.predict("first.jpg")
    service.predict("second.jpg")

    assert len(instances) == 1
    assert instances[0].calls == [Path("first.jpg"), Path("second.jpg")]


def test_service_wraps_prediction_failure() -> None:
    class BrokenPredictor:
        def __init__(self, config: InferenceConfig) -> None:
            pass

        def predict(self, target_frame: Path) -> None:
            raise ValueError("boom")

    service = InferenceService(InferenceConfig(), predictor_factory=BrokenPredictor)

    with pytest.raises(InferenceAPIError, match="Local inference failed") as error:
        service.predict("frame.jpg")

    assert isinstance(error.value.__cause__, ValueError)


def test_api_import_keeps_package_root_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "local_extraction.inference" or name.startswith("local_extraction.inference."):
            monkeypatch.delitem(sys.modules, name, raising=False)
        if name in {"torch", "torchvision", "ultralytics"}:
            monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("local_extraction.inference.api")

    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "ultralytics" not in sys.modules
