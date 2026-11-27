import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory


def transcribe(source: Path, model_name: str, language: str | None) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found. {source}")

    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(source), language=language)
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize(text: str, voice: str, destination: Path) -> Path:
    from edge_tts import Communicate

    if destination.suffix.lower() != ".mp3":
        raise ValueError("Choose an output filename ending in .mp3.")

    async def write_audio(path):
        await asyncio.wait_for(Communicate(text, voice).save(str(path)), timeout=120)

    with TemporaryDirectory(dir=destination.parent) as temporary:
        draft = Path(temporary) / "speech.mp3"
        try:
            asyncio.run(write_audio(draft))
        except TimeoutError as error:
            raise RuntimeError(
                "Edge TTS timed out. Try again or use a shorter script."
            ) from error
        if not draft.stat().st_size:
            raise RuntimeError("Edge TTS returned an empty audio file.")
        draft.replace(destination)
    return destination
