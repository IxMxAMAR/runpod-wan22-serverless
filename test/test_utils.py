"""Tests for handler utility functions."""
import json
import os
import tempfile
import pytest

from handler.utils import (
    validate_loras,
    calculate_frames,
    resolve_resolution,
    generate_seed,
)


class TestValidateLoras:
    """Tests for LoRA file validation against network volume."""

    def test_valid_lora_exists(self, tmp_path):
        """LoRA file exists on volume — returns resolved path."""
        lora_dir = tmp_path / "models" / "loras"
        lora_dir.mkdir(parents=True)
        (lora_dir / "test-lora.safetensors").touch()

        result = validate_loras(
            [{"name": "test-lora", "strength": 1.0}],
            str(tmp_path),
        )
        assert result == [{"name": "test-lora.safetensors", "strength": 1.0}]

    def test_valid_lora_in_subdirectory(self, tmp_path):
        """LoRA in subdirectory — name includes relative path."""
        lora_dir = tmp_path / "models" / "loras" / "wan"
        lora_dir.mkdir(parents=True)
        (lora_dir / "FusionX.safetensors").touch()

        result = validate_loras(
            [{"name": "wan/FusionX", "strength": 0.8}],
            str(tmp_path),
        )
        assert result == [{"name": "wan/FusionX.safetensors", "strength": 0.8}]

    def test_lora_not_found_raises(self, tmp_path):
        """Missing LoRA file raises ValueError with clear message."""
        lora_dir = tmp_path / "models" / "loras"
        lora_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="LoRA 'nonexistent' not found"):
            validate_loras(
                [{"name": "nonexistent", "strength": 1.0}],
                str(tmp_path),
            )

    def test_lora_name_with_extension(self, tmp_path):
        """Name already has .safetensors extension — works fine."""
        lora_dir = tmp_path / "models" / "loras"
        lora_dir.mkdir(parents=True)
        (lora_dir / "test.safetensors").touch()

        result = validate_loras(
            [{"name": "test.safetensors", "strength": 1.0}],
            str(tmp_path),
        )
        assert result == [{"name": "test.safetensors", "strength": 1.0}]

    def test_empty_loras_list(self, tmp_path):
        """Empty list — returns empty list."""
        result = validate_loras([], str(tmp_path))
        assert result == []

    def test_multiple_loras(self, tmp_path):
        """Multiple LoRAs — validates all."""
        lora_dir = tmp_path / "models" / "loras"
        lora_dir.mkdir(parents=True)
        (lora_dir / "a.safetensors").touch()
        (lora_dir / "b.safetensors").touch()

        result = validate_loras(
            [
                {"name": "a", "strength": 1.0},
                {"name": "b", "strength": 0.5},
            ],
            str(tmp_path),
        )
        assert len(result) == 2

    def test_default_strength(self, tmp_path):
        """Missing strength defaults to 1.0."""
        lora_dir = tmp_path / "models" / "loras"
        lora_dir.mkdir(parents=True)
        (lora_dir / "test.safetensors").touch()

        result = validate_loras(
            [{"name": "test"}],
            str(tmp_path),
        )
        assert result == [{"name": "test.safetensors", "strength": 1.0}]


class TestCalculateFrames:
    """Tests for duration-to-frame calculation."""

    def test_t2v_frames(self):
        """T2V: 5 seconds * 16 = 80 frames."""
        assert calculate_frames(5, "t2v") == 80

    def test_i2v_frames(self):
        """I2V: 5 seconds * 15 = 75 frames."""
        assert calculate_frames(5, "i2v") == 75

    def test_short_duration(self):
        """1 second T2V = 16 frames."""
        assert calculate_frames(1, "t2v") == 16

    def test_fractional_duration(self):
        """2.5 seconds T2V = 40 frames."""
        assert calculate_frames(2.5, "t2v") == 40

    def test_zero_raises(self):
        """Zero duration raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            calculate_frames(0, "t2v")

    def test_negative_raises(self):
        """Negative duration raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            calculate_frames(-1, "t2v")


class TestResolveResolution:
    """Tests for resolution parsing from width/height or aspect ratio."""

    def test_explicit_width_height(self):
        """Explicit width and height passed through."""
        result = resolve_resolution({"width": 832, "height": 480})
        assert result == (832, 480)

    def test_aspect_ratio_16_9_horizontal(self):
        """16:9 horizontal at default scale."""
        w, h = resolve_resolution({"aspect_ratio": "16:9"})
        assert w > h
        assert w / h == pytest.approx(16 / 9, abs=0.1)

    def test_aspect_ratio_9_16_vertical(self):
        """9:16 vertical."""
        w, h = resolve_resolution({"aspect_ratio": "9:16"})
        assert h > w

    def test_aspect_ratio_1_1(self):
        """1:1 square."""
        w, h = resolve_resolution({"aspect_ratio": "1:1"})
        assert w == h

    def test_default_resolution(self):
        """No input returns default 832x480."""
        result = resolve_resolution({})
        assert result == (832, 480)

    def test_none_input(self):
        """None input returns default."""
        result = resolve_resolution(None)
        assert result == (832, 480)


class TestGenerateSeed:
    """Tests for seed generation."""

    def test_explicit_seed(self):
        """Explicit seed returned as-is."""
        assert generate_seed(42) == 42

    def test_none_generates_random(self):
        """None seed generates a random integer."""
        seed = generate_seed(None)
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    def test_random_seeds_differ(self):
        """Two random seeds are (almost certainly) different."""
        s1 = generate_seed(None)
        s2 = generate_seed(None)
        assert s1 != s2
