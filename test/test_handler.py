"""Tests for the RunPod handler request routing and validation."""
import json
import pytest

from handler.handler import validate_input


class TestValidateInput:
    """Test input validation before processing."""

    def test_template_mode_valid(self):
        inp = {
            "template": "t2v-standard",
            "params": {"prompt": "a cat"},
        }
        mode, data = validate_input(inp)
        assert mode == "template"
        assert data["params"]["prompt"] == "a cat"

    def test_workflow_mode_valid(self):
        inp = {
            "workflow": {"1": {"class_type": "CLIPTextEncode", "inputs": {}}},
        }
        mode, data = validate_input(inp)
        assert mode == "workflow"

    def test_template_mode_missing_prompt_raises(self):
        inp = {
            "template": "t2v-standard",
            "params": {"duration": 5},
        }
        with pytest.raises(ValueError, match="'prompt' is required"):
            validate_input(inp)

    def test_template_mode_missing_params_raises(self):
        inp = {"template": "t2v-standard"}
        with pytest.raises(ValueError, match="'params' is required"):
            validate_input(inp)

    def test_neither_mode_raises(self):
        inp = {"something": "else"}
        with pytest.raises(ValueError, match="must contain either 'template' or 'workflow'"):
            validate_input(inp)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="must contain either"):
            validate_input({})

    def test_workflow_mode_with_images(self):
        inp = {
            "workflow": {"1": {"class_type": "Test", "inputs": {}}},
            "images": [{"name": "test.png", "image": "base64data"}],
        }
        mode, data = validate_input(inp)
        assert mode == "workflow"
        assert len(data["images"]) == 1

    def test_i2v_template_requires_input_image(self):
        inp = {
            "template": "i2v-standard",
            "params": {"prompt": "test"},
        }
        with pytest.raises(ValueError, match="'input_image' is required for I2V"):
            validate_input(inp)

    def test_i2v_template_with_image_valid(self):
        inp = {
            "template": "i2v-standard",
            "params": {
                "prompt": "test",
                "input_image": "base64imagedata",
            },
        }
        mode, data = validate_input(inp)
        assert mode == "template"
