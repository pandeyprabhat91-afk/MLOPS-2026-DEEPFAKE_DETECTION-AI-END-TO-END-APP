"""Shared pytest fixtures."""

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def client():
    """Test client with model loading mocked out."""
    mock_model = MagicMock()
    with patch("backend.app.model_loader.load_model"):
        with patch("backend.app.model_loader._model", mock_model):
            with patch(
                "backend.app.model_loader._current_version", "deepfake/Production/1"
            ):
                from fastapi.testclient import TestClient

                from backend.app.main import app

                return TestClient(app)


@pytest.fixture
def sample_mp4_bytes():
    """Minimal bytes with .mp4 filename for upload tests."""
    return b"\x00\x00\x00\x08ftyp" + b"\x00" * 100


@pytest.fixture
def mock_frames_tensor():
    """Random frames tensor matching model input shape."""
    return torch.randn(30, 3, 224, 224)
