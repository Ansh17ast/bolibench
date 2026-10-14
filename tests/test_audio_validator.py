"""Unit tests for AudioValidator."""

import pytest
from app.services.audio_validator import AudioValidationError, AudioValidator


def test_audio_validator_valid_wav(valid_wav_bytes):
    """Test valid WAV file passes validation."""
    validator = AudioValidator(max_size_bytes=5 * 1024 * 1024)
    ext, content_type = validator.validate("test_sample.wav", valid_wav_bytes)

    assert ext == ".wav"
    assert content_type == "audio/wav"


def test_audio_validator_valid_mp3():
    """Test valid MP3 file with ID3 tag passes validation."""
    validator = AudioValidator()
    mp3_bytes = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90d" + b"\x00" * 100
    ext, content_type = validator.validate("recording.mp3", mp3_bytes)

    assert ext == ".mp3"
    assert content_type == "audio/mpeg"


def test_audio_validator_valid_ogg():
    """Test valid OGG file passes validation."""
    validator = AudioValidator()
    ogg_bytes = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 50
    ext, content_type = validator.validate("audio.ogg", ogg_bytes)

    assert ext == ".ogg"
    assert content_type == "audio/ogg"


def test_audio_validator_valid_flac():
    """Test valid FLAC file passes validation."""
    validator = AudioValidator()
    flac_bytes = b"fLaC\x00\x00\x00\x22" + b"\x00" * 50
    ext, content_type = validator.validate("clip.flac", flac_bytes)

    assert ext == ".flac"
    assert content_type == "audio/flac"


def test_audio_validator_missing_filename(valid_wav_bytes):
    """Test validation fails when filename is missing or empty."""
    validator = AudioValidator()
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate("", valid_wav_bytes)
    assert exc_info.value.error_type == "MISSING_FILENAME"


def test_audio_validator_unsupported_extension(valid_wav_bytes):
    """Test validation fails when extension is unsupported."""
    validator = AudioValidator()
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate("document.pdf", valid_wav_bytes)
    assert exc_info.value.error_type == "UNSUPPORTED_EXTENSION"


def test_audio_validator_empty_file():
    """Test validation fails on 0-byte file."""
    validator = AudioValidator()
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate("empty.wav", b"")
    assert exc_info.value.error_type == "EMPTY_FILE"


def test_audio_validator_file_too_large(valid_wav_bytes):
    """Test validation fails when file exceeds size ceiling."""
    validator = AudioValidator(max_size_bytes=100)  # Low ceiling for testing
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate("sample.wav", valid_wav_bytes)
    assert exc_info.value.error_type == "FILE_TOO_LARGE"


def test_audio_validator_invalid_wav_signature():
    """Test validation fails when file has .wav extension but invalid header bytes."""
    validator = AudioValidator()
    corrupt_bytes = b"CORRUPT_HEADER_BYTES_NOT_A_WAV_FILE"
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate("corrupt.wav", corrupt_bytes)
    assert exc_info.value.error_type == "INVALID_AUDIO_SIGNATURE"


def test_audio_validator_duration_exceeded(long_wav_bytes):
    """Test validation fails when audio duration exceeds 60 seconds."""
    validator = AudioValidator()
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate("long_call.wav", long_wav_bytes)
    assert exc_info.value.error_type == "DURATION_EXCEEDED"
    assert "exceeds the maximum allowed limit of 60s" in exc_info.value.message
