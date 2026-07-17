from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.voice_input import VoiceInputConfig, VoiceInputError, VoiceInputService


class FakeTranscriber:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe_file(self, audio_path: Path) -> str:
        self.paths.append(audio_path)
        return 'open the file tree'


class FakeProcess:
    def __init__(self, returncode_after_stop=-2) -> None:
        self.returncode = None
        self.returncode_after_stop = returncode_after_stop
        self.signals: list[int] = []
        self.killed = False

    def poll(self):  # noqa: ANN201
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = self.returncode_after_stop

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def communicate(self, timeout=None):  # noqa: ANN001, ANN201
        return ('', '')


def test_voice_input_start_stop_records_and_transcribes_with_arecord() -> None:
    transcriber = FakeTranscriber()
    service = VoiceInputService(
        VoiceInputConfig(duration_seconds=1, sample_rate=16000),
        transcriber=transcriber,
    )
    calls = []
    process = FakeProcess()

    def fake_popen(command, **kwargs):  # noqa: ANN001
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b'RIFF' + (b'0' * 128))
        return process

    with patch('src.voice_input.shutil.which', return_value='/usr/bin/arecord'):
        with patch('src.voice_input.subprocess.Popen', side_effect=fake_popen):
            recording = service.start_recording()
            text = recording.stop_and_transcribe()

    assert text == 'open the file tree'
    assert calls[0][0][:8] == ['/usr/bin/arecord', '-q', '-f', 'S16_LE', '-r', '16000', '-c', '1']
    assert '-d' not in calls[0][0]
    assert process.signals
    assert transcriber.paths


def test_voice_input_accepts_arecord_returncode_one_when_wav_exists() -> None:
    transcriber = FakeTranscriber()
    service = VoiceInputService(transcriber=transcriber)
    process = FakeProcess(returncode_after_stop=1)

    def fake_popen(command, **kwargs):  # noqa: ANN001
        Path(command[-1]).write_bytes(b'RIFF' + (b'0' * 128))
        return process

    with patch('src.voice_input.shutil.which', return_value='/usr/bin/arecord'):
        with patch('src.voice_input.subprocess.Popen', side_effect=fake_popen):
            recording = service.start_recording()
            text = recording.stop_and_transcribe()

    assert text == 'open the file tree'
    assert transcriber.paths


def test_voice_input_reports_missing_arecord() -> None:
    service = VoiceInputService(transcriber=FakeTranscriber())
    with patch('src.voice_input.shutil.which', return_value=None):
        with pytest.raises(VoiceInputError, match='arecord'):
            service.start_recording()
