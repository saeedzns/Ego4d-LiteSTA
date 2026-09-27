import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_model_artifacts: requires external model/checkpoint artifacts not stored in Git",
    )
