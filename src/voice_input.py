from __future__ import annotations

from dataclasses import dataclass
import os
import signal
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Protocol


class VoiceInputError(RuntimeError):
    """Raised when local voice capture or transcription cannot complete."""


class Transcriber(Protocol):
    def transcribe_file(self, audio_path: Path) -> str:
        ...


class VoiceRecording:
    def __init__(
        self,
        process: subprocess.Popen[str],
        audio_path: Path,
        temp_dir: tempfile.TemporaryDirectory[str],
        transcriber: Transcriber,
    ) -> None:
        self.process = process
        self.audio_path = audio_path
        self.temp_dir = temp_dir
        self.transcriber = transcriber
        self._closed = False

    def stop_and_transcribe(self) -> str:
        if self._closed:
            raise VoiceInputError('Voice recording has already been stopped.')
        self._closed = True
        try:
            self._stop_process()
            text = self.transcriber.transcribe_file(self.audio_path)
        finally:
            self.temp_dir.cleanup()
        text = ' '.join(text.split())
        if not text:
            raise VoiceInputError('No speech was detected.')
        return text

    def cancel(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stop_process()
        finally:
            self.temp_dir.cleanup()

    def _stop_process(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        try:
            _stdout, stderr = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.communicate()
            raise VoiceInputError('Timed out while stopping voice recording.') from exc
        if self.process.returncode not in (0, 1, -15, -2, 143, 130):
            detail = (stderr or '').strip()
            raise VoiceInputError(f'Voice recording failed: {detail or self.process.returncode}')
        if self.audio_path.exists() and self.audio_path.stat().st_size > 44:
            return
        detail = (stderr or '').strip()
        raise VoiceInputError(f'Voice recording produced no usable audio: {detail or self.process.returncode}')



@dataclass(frozen=True)
class VoiceInputConfig:
    duration_seconds: int = 5
    sample_rate: int = 16000
    model: str = 'Systran/faster-whisper-base'
    device: str = 'cpu'
    compute_type: str = 'int8'
    language: str | None = None
    local_files_only: bool = True

    @classmethod
    def from_env(cls) -> 'VoiceInputConfig':
        return cls(
            duration_seconds=_env_int('CLAWED_VOICE_DURATION_SECONDS', 5),
            sample_rate=_env_int('CLAWED_VOICE_SAMPLE_RATE', 16000),
            model=os.environ.get('CLAWED_VOICE_MODEL', 'Systran/faster-whisper-base'),
            device=os.environ.get('CLAWED_VOICE_DEVICE', 'cpu'),
            compute_type=os.environ.get('CLAWED_VOICE_COMPUTE_TYPE', 'int8'),
            language=os.environ.get('CLAWED_VOICE_LANGUAGE') or None,
            local_files_only=_env_bool('CLAWED_VOICE_LOCAL_FILES_ONLY', True),
        )


class FasterWhisperTranscriber:
    def __init__(self, config: VoiceInputConfig) -> None:
        self.config = config
        self._model = None

    def transcribe_file(self, audio_path: Path) -> str:
        model = self._load_model()
        kwargs: dict[str, object] = {
            'beam_size': 1,
            'vad_filter': True,
        }
        if self.config.language:
            kwargs['language'] = self.config.language
        segments, _info = model.transcribe(str(audio_path), **kwargs)
        text = ' '.join(segment.text.strip() for segment in segments if segment.text.strip())
        return ' '.join(text.split())

    def _load_model(self):  # noqa: ANN202
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:  # pragma: no cover - depends on optional install
                raise VoiceInputError(
                    'faster-whisper is not installed in the ClawedCode environment.'
                ) from exc
            try:
                self._model = WhisperModel(
                    self.config.model,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    local_files_only=self.config.local_files_only,
                )
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                raise VoiceInputError(
                    f'Unable to load voice model {self.config.model!r}: {exc}'
                ) from exc
        return self._model


class VoiceInputService:
    def __init__(
        self,
        config: VoiceInputConfig | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.config = config or VoiceInputConfig.from_env()
        self.transcriber = transcriber or FasterWhisperTranscriber(self.config)

    def capture_and_transcribe(self) -> str:
        recording = self.start_recording()
        try:
            return recording.stop_and_transcribe()
        except Exception:
            recording.cancel()
            raise

    def start_recording(self) -> VoiceRecording:
        arecord = shutil.which('arecord')
        if arecord is None:
            raise VoiceInputError('arecord is not installed; install alsa-utils to use voice input.')
        temp_dir = tempfile.TemporaryDirectory(prefix='clawed-voice-')
        audio_path = Path(temp_dir.name) / 'voice.wav'
        command = [
            arecord,
            '-q',
            '-f',
            'S16_LE',
            '-r',
            str(self.config.sample_rate),
            '-c',
            '1',
            str(audio_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            temp_dir.cleanup()
            raise VoiceInputError(f'Unable to start voice recording: {exc}') from exc
        return VoiceRecording(process, audio_path, temp_dir, self.transcriber)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise VoiceInputError(f'{name} must be an integer') from exc
    if value < 1:
        raise VoiceInputError(f'{name} must be >= 1')
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}
