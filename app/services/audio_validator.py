"""Audio validation service for BoliBench.

Validates audio files against Gnani Prisma STT requirements:
- Containers: WAV, MP3, OGG, FLAC, AAC, M4A
- Size: Up to configured maximum (default 25 MB)
- Duration: Maximum 60 seconds (ideal <= 30 seconds)
"""

import io
import wave
from typing import Dict, Optional, Set, Tuple


class AudioValidationError(Exception):
    """Exception raised when an audio file fails validation."""

    def __init__(self, message: str, error_type: str = "INVALID_AUDIO_FILE", details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}


class AudioValidator:
    """Validates audio files for format, size, header integrity, and duration."""

    SUPPORTED_EXTENSIONS: Set[str] = {
        ".wav",
        ".mp3",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
    }

    # Live evaluation strictly restricts uploads to formats with verifiable duration
    # to protect owner credits from unvalidated long-audio consumption.
    LIVE_SUPPORTED_EXTENSIONS: Set[str] = {
        ".wav",
    }

    CONTENT_TYPE_MAP: Dict[str, str] = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
    }

    MAX_DURATION_SECONDS: float = 60.0
    IDEAL_DURATION_SECONDS: float = 30.0

    def __init__(self, max_size_bytes: int = 26_214_400):
        self.max_size_bytes = max_size_bytes

    def validate(self, filename: Optional[str], content: bytes) -> Tuple[str, str]:
        """Validates filename, size, header magic bytes, and duration.

        Returns:
            Tuple of (extension, content_type)
        """
        # 1. Filename validation
        if not filename or not filename.strip():
            raise AudioValidationError(
                "Audio filename must not be empty.",
                error_type="MISSING_FILENAME",
            )

        clean_filename = filename.strip().lower()
        ext = None
        for supported_ext in self.SUPPORTED_EXTENSIONS:
            if clean_filename.endswith(supported_ext):
                ext = supported_ext
                break

        if not ext:
            raise AudioValidationError(
                f"Unsupported audio format '{filename}'. "
                f"Supported formats are: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}",
                error_type="UNSUPPORTED_EXTENSION",
                details={"filename": filename, "supported": list(self.SUPPORTED_EXTENSIONS)},
            )

        # 2. File size validation
        size_bytes = len(content)
        if size_bytes == 0:
            raise AudioValidationError(
                "Uploaded audio file is empty (0 bytes).",
                error_type="EMPTY_FILE",
            )

        if size_bytes > self.max_size_bytes:
            max_mb = self.max_size_bytes / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise AudioValidationError(
                f"Audio file size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.2f} MB.",
                error_type="FILE_TOO_LARGE",
                details={"size_bytes": size_bytes, "max_size_bytes": self.max_size_bytes},
            )

        # 3. Header / Magic Bytes Validation
        self._validate_magic_bytes(ext, content)

        # 4. Duration Check (for WAV files via wave standard library)
        if ext == ".wav":
            self._validate_wav_duration(content)

        content_type = self.CONTENT_TYPE_MAP.get(ext, "application/octet-stream")
        return ext, content_type

    def validate_live_audio(self, filename: Optional[str], content: bytes) -> Tuple[str, str]:
        """Validates audio strictly for live evaluation where provider credits are at stake.

        Restrictions:
        - Only formats with verifiable duration (WAV format via wave header parsing)
          are accepted for live evaluation to guarantee duration limits (<= 60s) before
          consuming provider credits.
        - Other containers (.mp3, .ogg, .flac, .aac, .m4a) do not have lightweight
          standard-library duration parsers and are restricted from the live route
          to protect owner credits.
        """
        ext, content_type = self.validate(filename, content)

        if ext not in self.LIVE_SUPPORTED_EXTENSIONS:
            raise AudioValidationError(
                f"Live evaluation currently restricts uploads to formats with verified duration ({', '.join(sorted(self.LIVE_SUPPORTED_EXTENSIONS))}) "
                f"to guarantee duration limits (<= {self.MAX_DURATION_SECONDS:.0f}s) before spending provider credits. "
                f"Received '{filename}' ({ext}). Please convert audio to WAV format for live evaluation, or use the single-audio transcription lab.",
                error_type="UNSUPPORTED_LIVE_FORMAT",
                details={
                    "filename": filename,
                    "extension": ext,
                    "supported_live_formats": list(self.LIVE_SUPPORTED_EXTENSIONS),
                },
            )

        # Enforce strict WAV duration check (fails if wave header is unparseable)
        self._validate_wav_duration(content, strict=True)
        return ext, content_type

    def _validate_magic_bytes(self, ext: str, content: bytes) -> None:
        """Inspects file signature / magic bytes."""
        if len(content) < 4:
            raise AudioValidationError(
                "Audio file header is too short or corrupt.",
                error_type="CORRUPT_AUDIO_HEADER",
            )

        if ext == ".wav":
            if not (content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WAVE"):
                raise AudioValidationError(
                    "Invalid WAV file header. File does not contain expected RIFF/WAVE signature.",
                    error_type="INVALID_AUDIO_SIGNATURE",
                )
        elif ext == ".ogg":
            if not content.startswith(b"OggS"):
                raise AudioValidationError(
                    "Invalid OGG file header. File does not start with 'OggS'.",
                    error_type="INVALID_AUDIO_SIGNATURE",
                )
        elif ext == ".flac":
            if not content.startswith(b"fLaC"):
                raise AudioValidationError(
                    "Invalid FLAC file header. File does not start with 'fLaC'.",
                    error_type="INVALID_AUDIO_SIGNATURE",
                )
        elif ext == ".mp3":
            # MP3 files usually begin with ID3v2 tag or MPEG audio sync frame 0xFFFB / 0xFFFA / 0xFFF3 / 0xFFF2
            has_id3 = content.startswith(b"ID3")
            has_sync = len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0
            if not (has_id3 or has_sync):
                raise AudioValidationError(
                    "Invalid MP3 file header. Missing ID3 tag or MPEG frame sync header.",
                    error_type="INVALID_AUDIO_SIGNATURE",
                )
        elif ext in {".m4a", ".aac"}:
            # M4A contains 'ftyp' atom in first 16 bytes, or AAC ADTS starts with 0xFFF1 / 0xFFF9
            has_ftyp = b"ftyp" in content[:16]
            has_adts = len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xF0) == 0xF0
            if not (has_ftyp or has_adts):
                raise AudioValidationError(
                    f"Invalid {ext.upper()} file header. Missing ftyp container or ADTS sync word.",
                    error_type="INVALID_AUDIO_SIGNATURE",
                )

    def _validate_wav_duration(self, content: bytes, strict: bool = False) -> None:
        """Parses WAV duration and enforces the 60s hard limit."""
        try:
            with wave.open(io.BytesIO(content), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                if rate > 0:
                    duration = frames / float(rate)
                    if duration > self.MAX_DURATION_SECONDS:
                        raise AudioValidationError(
                            f"Audio duration ({duration:.1f}s) exceeds the maximum allowed limit of {self.MAX_DURATION_SECONDS:.0f}s. "
                            f"(Ideal duration is <= {self.IDEAL_DURATION_SECONDS:.0f}s).",
                            error_type="DURATION_EXCEEDED",
                            details={"duration_seconds": duration, "max_allowed": self.MAX_DURATION_SECONDS},
                        )
        except wave.Error as exc:
            if strict:
                raise AudioValidationError(
                    f"Unable to parse WAV duration from header: {str(exc)}. Audio must be standard PCM WAV.",
                    error_type="INVALID_WAV_FORMAT",
                )
            # Non-standard WAV or compressed chunk; defer full duration check to API in non-strict mode
            pass
