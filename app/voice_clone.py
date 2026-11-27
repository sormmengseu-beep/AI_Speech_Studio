import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from unicodedata import normalize

import numpy as np
import soundfile as sf

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
)
KHMER_LANGUAGE = "Khmer"
KHMER_MODEL_ID = "muyleanging/VoxPocket-TTS-Khmer"
KHMER_MODEL_REVISION = "7c4390ed1813b56a5b4c36d5ec29ee6d12eccbcb"
KHMER_MODEL_FILES = (
    "audiovae.pth",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
QWEN_LANGUAGES = (
    "English",
    "Chinese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)
CLONE_LANGUAGES = ("English", KHMER_LANGUAGE, *QWEN_LANGUAGES[1:])


@dataclass(frozen=True)
class VoiceModel:
    name: str
    languages: tuple[str, ...]
    sample_rate: str
    download_size: str
    reference_length: str
    requires_transcript: bool = False


VOICE_MODELS = {
    MODEL_ID: VoiceModel("Qwen3-TTS 0.6B", QWEN_LANGUAGES, "24 kHz", "2.5 GB", "5–15 seconds"),
    KHMER_MODEL_ID: VoiceModel(
        "VoxPocket · Khmer", (KHMER_LANGUAGE,), "44.1 kHz", "3.6 GB", "4–9 seconds",
        requires_transcript=True,
    ),
}


def resolve_voice_model_id(language: str, model_id: str | None = None) -> str:
    if model_id is None:
        model_id = KHMER_MODEL_ID if language == KHMER_LANGUAGE else MODEL_ID
    model = VOICE_MODELS.get(model_id)
    if model is None:
        raise ValueError("Choose a supported voice clone model.")
    if language not in model.languages:
        raise ValueError(f"{model.name} does not support {language}. Choose a supported language.")
    return model_id


def _model_directory(model_id, revision, files):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        cached = Path(
            snapshot_download(model_id, revision=revision, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        cached = None

    if cached is not None and all((cached / name).is_file() for name in files):
        return cached

    return Path(
        snapshot_download(model_id, revision=revision, allow_patterns=list(files))
    )


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install the voice clone dependencies from requirements-voice.txt."
        ) from error

    torch.set_num_threads(min(os.cpu_count() or 1, 6))
    return Qwen3TTSModel.from_pretrained(
        str(_model_directory(MODEL_ID, MODEL_REVISION, MODEL_FILES)),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    )


@lru_cache(maxsize=1)
def _load_khmer_model():
    try:
        import torch
        from voxcpm import VoxCPM
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install the Khmer voice dependencies from requirements-voice.txt."
        ) from error

    torch.set_num_threads(min(os.cpu_count() or 1, 6))
    directory = _model_directory(
        KHMER_MODEL_ID, KHMER_MODEL_REVISION, KHMER_MODEL_FILES
    )
    return VoxCPM.from_pretrained(
        str(directory), device="cpu", optimize=False, load_denoiser=False
    )


def _ffmpeg_dll_directory():
    if os.name == "nt":
        directory = Path(sys.prefix) / "ffmpeg-shared" / "bin"
        if directory.is_dir():
            return os.add_dll_directory(directory)
    return nullcontext()


def _khmer_sections(text):
    sentences = re.split(r"(?<=[។！？!?])\s*|(?<=\.)\s+|\n+", text)
    section = []
    for sentence in filter(None, map(str.strip, sentences)):
        if section and (len(section) == 3 or len(" ".join([*section, sentence])) > 380):
            yield " ".join(section)
            section.clear()
        section.append(sentence)
    if section:
        yield " ".join(section)


def _generate_khmer(reference, transcript, text):
    prompt_text = normalize("NFC", transcript) + " "
    with _ffmpeg_dll_directory():
        model = _load_khmer_model()
        sample_rate = model.tts_model.sample_rate
        pause = np.zeros(round(0.12 * sample_rate), dtype=np.float32)
        clips = []
        for section in _khmer_sections(normalize("NFC", text)):
            if clips:
                clips.append(pause)
            clips.append(
                model.generate(
                    text=section,
                    prompt_text=prompt_text,
                    prompt_wav_path=str(reference),
                    cfg_value=2.0,
                    inference_timesteps=10,
                )
            )
    return np.concatenate(clips), sample_rate


def _read_reference(reference: Path):
    with sf.SoundFile(reference) as file:
        sample_rate = file.samplerate
        if not 3 <= file.frames / sample_rate <= 30:
            raise ValueError("Choose a reference recording between 3 and 30 seconds.")
        samples = file.read(dtype="float32", always_2d=True)

    mono = samples.mean(axis=1)
    if not np.isfinite(mono).all() or np.max(np.abs(mono), initial=0) < 0.0001:
        raise ValueError("Choose a reference recording with audible speech.")
    return mono, sample_rate


def _render_voice(
    reference: Path,
    reference_text: str,
    text: str,
    language: str,
    destination: Path,
    model_id: str | None = None,
) -> Path:
    if destination.suffix.lower() != ".wav":
        raise ValueError("Choose an output filename ending in .wav.")
    if destination.resolve() == reference.resolve():
        raise ValueError(
            "Choose a different output file to keep the reference recording."
        )
    if not text.strip():
        raise ValueError("Enter the text you want the voice to speak.")

    model_id = resolve_voice_model_id(language, model_id)
    transcript = reference_text.strip()
    if VOICE_MODELS[model_id].requires_transcript and not transcript:
        raise ValueError(
            "Enter the exact Khmer words spoken in the reference recording."
        )

    reference_audio = _read_reference(reference)
    if model_id == KHMER_MODEL_ID:
        _load_model.cache_clear()
        audio, sample_rate = _generate_khmer(reference, transcript, text.strip())
    else:
        _load_khmer_model.cache_clear()
        model = _load_model()
        waveforms, sample_rate = model.generate_voice_clone(
            text=text.strip(),
            language=language,
            ref_audio=reference_audio,
            ref_text=transcript or None,
            x_vector_only_mode=not transcript,
            non_streaming_mode=True,
        )
        audio = waveforms[0]
    peak = np.max(np.abs(audio), initial=0)
    if not np.isfinite(peak) or peak == 0:
        raise RuntimeError(
            "The voice model returned no usable audio. Try a different clip."
        )
    audio = audio * min(2.0, 0.89125 / peak)

    with TemporaryDirectory(dir=destination.parent) as temporary:
        draft = Path(temporary) / "cloned-voice.wav"
        sf.write(draft, audio, sample_rate, subtype="PCM_24")
        draft.replace(destination)
    return destination


@lru_cache(maxsize=1)
def _voice_worker():
    # Keep PyTorch's native runtime outside the Faster-Whisper desktop process.
    return ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))


def clone_voice(
    reference: Path,
    reference_text: str,
    text: str,
    language: str,
    destination: Path,
    model_id: str | None = None,
) -> Path:
    worker = _voice_worker()
    try:
        return worker.submit(
            _render_voice, reference, reference_text, text, language, destination, model_id
        ).result()
    except BrokenProcessPool as error:
        _voice_worker.cache_clear()
        worker.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(
            "The voice engine exited. Try generating the clip again."
        ) from error
