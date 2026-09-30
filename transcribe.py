"""
Audio Transcription CLI (faster-whisper)
========================================

Transcribe audio/video files (mp3, flac, wav, m4a, ogg, opus, mp4, ...) into
text, subtitle and JSON files using a local Whisper model served by
CTranslate2.

WHY FASTER-WHISPER
------------------
* Same Whisper weights/accuracy as OpenAI Whisper, but the CTranslate2 runtime
  makes CPU inference roughly 4x faster and drops the ~2.5 GB torch
  dependency. ``int8`` quantization is used on CPU, ``float16`` on GPU.
* Any container/codec is decoded through PyAV, so no manual ``ffmpeg``
  conversion step is needed for the input file.
* The bundled Silero VAD skips silence: faster, and no hallucinated text
  inside long gaps.
* Segment (and optionally word) timestamps come for free -> SRT/VTT subtitles.

MODEL SIZE vs SPEED (14-core CPU, 37m51s of audio, int8)
-------------------------------------------------------
Time to decode that same ~38-minute file. The `tiny` and `distil-large-v3`
rows were measured on this machine; the rest are scaled estimates.

    model             38 min of audio   notes
    tiny                1 min (38x)     rough, smoke tests only
    base               ~2 min           okay
    small              ~4 min           good quality/speed balance
    distil-large-v3     8m38s (4.4x)    near large-v3 accuracy  <- DEFAULT (English)
    medium            ~13 min           solid, multilingual
    large-v3          ~50 min           best accuracy, multilingual

DEFAULT MODEL
-------------
``distil-large-v3`` keeps near large-v3 accuracy while being ~6x faster than
large-v3. It is English-only, so for it the script rejects ``--task translate``
and non-English ``--language`` values and points you at ``--model large-v3``
instead.

MODEL CACHE
-----------
The weights are kept inside the checkout (``<repo>/models/<model-name>/``,
gitignored) instead of the shared HuggingFace cache, so a clone owns its models.
Before any audio is decoded the script checks that folder and, when the model is
missing, downloads it up front -- ``distil-large-v3`` is ~1.5 GB, ``tiny`` ~75 MB
-- reporting where it landed; later runs find the folder and need no network. A
model that already sits in ``~/.cache/huggingface`` is copied rather than fetched
again. ``--hf-cache`` keeps models in the shared cache instead, ``--model-dir``
moves the folder, and ``--prefetch-model`` downloads and exits (useful in
CI/Docker) so the first real run cannot fail on a slow connection.

OUTPUT
------
Every input gets its own folder inside the output directory (default
``./transcripts/``), so all artifacts of one recording stay together::

    transcripts/
      Andrew Ng - The Biggest Opportunities in AI/
        Andrew Ng - The Biggest Opportunities in AI.txt    # plain text
        Andrew Ng - The Biggest Opportunities in AI.srt    # subtitles
        Andrew Ng - The Biggest Opportunities in AI.json   # metadata + segments
        Andrew Ng - The Biggest Opportunities in AI.md     # notes you keep too

``--flat`` writes straight into the output directory instead (the old layout: one
big pile of files). Input files are never modified.

Long files are checkpointed: every ``--checkpoint-seconds`` (default 60) the
segments decoded so far are flushed to ``<stem>.partial.<ext>`` in that same
folder, and Ctrl-C keeps that checkpoint instead of throwing the work away. The
real names are written once the file is finished and the checkpoints are deleted,
so a half-decoded file can never be mistaken for a finished transcript (and
``--skip-existing`` always re-does an interrupted file).

PROGRESS
--------
Progress is written to *stderr* (stdout stays clean, so redirect it freely): a
live single line on a terminal, and in a redirected log (``nohup``, ``>run.log``)
one complete line at every 10% plus at least every ``--progress-seconds``
(default 15; ``0`` = only the 10% lines)::

    progress  42.4%  16:03/37:51  elapsed 03:52  eta 05:07  4.19x

``-q`` prints only the final summary; ``--log-progress`` hands progress over to
the library's own tqdm bar.

PICKING FILES
-------------
Passing a file path transcribes just that file; a directory is expanded to every
media file inside it. To avoid retyping an awkward name - or to skip one
recording in a batch - the discovered list can be filtered::

    --only TEXT      keep files whose name contains TEXT (case-insensitive,
                     wildcards such as '*lecture*' work, repeatable)
    --exclude TEXT   drop files whose name contains TEXT (repeatable)
    --pick N         keep the Nth selected file(s) as numbered by --list-files
                     ('3', '1,4' and '2-5' are accepted)
    --list-files     print the numbered files that would be transcribed, then exit

``--only`` runs first, then ``--exclude``, then ``--pick`` numbers the survivors,
so the numbers always match the ``--list-files`` output::

    python transcribe.py ytaudio/ --list-files           # 1. ... 2. ...
    python transcribe.py ytaudio/ --only "Andrew Ng"     # only that recording
    python transcribe.py ytaudio/ --pick 2               # the second one
    python transcribe.py ytaudio/ --exclude trailer      # everything but that

USAGE
-----
    python transcribe.py AUDIO_OR_DIR [AUDIO_OR_DIR ...] [options]

    python transcribe.py ytaudio/                       # whole folder
    python transcribe.py talk.mp3 -f txt,srt,json
    python transcribe.py lecture.wav -m large-v3 -l de
    python transcribe.py ytaudio/ --only "Andrew Ng"    # just one file of the folder
    python transcribe.py ytaudio/ --pick 1              # the 1st from --list-files
    python transcribe.py ytaudio/ --skip-existing       # resumable batch
    python transcribe.py ytaudio/ --flat                # old one-folder layout
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

#: Repository root: the folder that holds this script. Downloaded models are
#: kept in ``<repo>/models`` so a checkout carries its own Whisper weights.
REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "distil-large-v3"
DEFAULT_OUTPUT_DIR = "transcripts"
#: Where models pulled from HuggingFace are stored (repo-local by default).
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
#: The one file every CTranslate2 model folder must contain.
MODEL_BIN = "model.bin"
DEFAULT_FORMATS = ("txt", "srt")
ALL_FORMATS = ("txt", "srt", "vtt", "json")
DEFAULT_CHECKPOINT_SECONDS = 60
DEFAULT_PROGRESS_SECONDS = 15.0

#: Model names faster-whisper understands (a local CTranslate2 directory also works).
KNOWN_MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2",
    "distil-large-v3", "distil-large-v3.5",
)

#: Extensions picked up when an input argument is a directory.
MEDIA_EXTENSIONS = frozenset(
    """
    .mp3 .flac .wav .m4a .aac .ogg .oga .opus .wma .aiff .aif .amr .ac3
    .mp4 .mkv .webm .mov .avi .m4v .flv .wmv .mpeg .mpg .ts
    """.split()
)

# Windows-illegal characters only; unicode (e.g. full-width punctuation) is kept.
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def human_duration(seconds: float) -> str:
    """``3725.4`` -> ``1h02m05s`` (or ``12m05s``)."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def clock(seconds: float) -> str:
    """``minutes:seconds`` label used for console output: ``mm:ss`` or ``h:mm:ss``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float, decimal_separator: str = ",") -> str:
    """Timestamp for subtitle files: ``HH:MM:SS<sep>mmm``."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{millis:03d}"


def sanitize_stem(name: str) -> str:
    """Make a file *stem* safe as a filename while preserving unicode letters."""
    stem = _ILLEGAL_FILENAME_CHARS.sub("_", name)
    stem = _WHITESPACE.sub(" ", stem).strip().strip(".")
    return stem or "transcript"


def is_english_only(model_name: str) -> bool:
    """True for checkpoints that cannot translate and only understand English."""
    lowered = model_name.lower()
    return lowered.startswith("distil-") or lowered.endswith(".en")


# ---------------------------------------------------------------------------
# Transcript data + output writers
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One decoded speech segment, decoupled from the library's own type."""

    start: float
    end: float
    text: str
    words: list[dict] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def as_dict(self) -> dict:
        data = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        if self.words is not None:
            data["words"] = [
                {
                    "start": round(word["start"], 3),
                    "end": round(word["end"], 3),
                    "word": word["word"],
                    "probability": round(word["probability"], 4),
                }
                for word in self.words
            ]
        if self.avg_logprob is not None:
            data["avg_logprob"] = round(self.avg_logprob, 4)
        if self.no_speech_prob is not None:
            data["no_speech_prob"] = round(self.no_speech_prob, 4)
        return data


def to_segment(raw_segment, keep_words: bool) -> Segment:
    """Convert a faster-whisper segment into our own lightweight :class:`Segment`."""
    words = None
    if keep_words:
        words = [
            {
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability,
            }
            for word in (getattr(raw_segment, "words", None) or [])
        ]
    return Segment(
        start=float(raw_segment.start),
        end=float(raw_segment.end),
        text=raw_segment.text.strip(),
        words=words,
        avg_logprob=getattr(raw_segment, "avg_logprob", None),
        no_speech_prob=getattr(raw_segment, "no_speech_prob", None),
    )


def render_txt(segments: list[Segment], with_timestamps: bool = False) -> str:
    lines = []
    for segment in segments:
        if not segment.text:
            continue
        lines.append(f"[{clock(segment.start)}] {segment.text}" if with_timestamps else segment.text)
    return "\n".join(lines) + "\n" if lines else ""


def render_srt(segments: list[Segment]) -> str:
    blocks = []
    for index, segment in enumerate((s for s in segments if s.text), start=1):
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{segment.text}\n"
        )
    return "\n".join(blocks)


def render_vtt(segments: list[Segment]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in (s for s in segments if s.text):
        blocks.append(
            f"{format_timestamp(segment.start, '.')} --> {format_timestamp(segment.end, '.')}\n"
            f"{segment.text}\n"
        )
    return "\n".join(blocks)


def render_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def build_json_payload(
    path: Path,
    args: argparse.Namespace,
    device: str,
    compute_type: str,
    info,
    segments: list[Segment],
) -> dict:
    """Rich, self-describing JSON: metadata plus every timed segment."""
    return {
        "source_file": str(path.resolve()),
        "model": args.model,
        "device": device,
        "compute_type": compute_type,
        "task": args.task,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "duration_seconds": round(float(info.duration), 3),
        "duration": human_duration(info.duration),
        "segment_count": len(segments),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "transcript": " ".join(s.text for s in segments if s.text).strip(),
        "segments": [segment.as_dict() for segment in segments],
    }


class ProgressReporter:
    """Percentage progress driven by the decoded segment stream.

    faster-whisper's progress API differs between releases (``progress_callback``
    in older versions, ``log_progress`` in 1.2.x), so progress is derived from how
    far into the file the finished segments reach: version-independent and free.

    On a terminal it redraws one live line. When stderr is redirected - a log
    file, ``nohup``, CI - it prints a complete line at every 10% boundary and at
    most every ``log_seconds``, so the log stays readable while always showing how
    far the current file has got.
    """

    def __init__(self, enabled: bool, log_seconds: float = DEFAULT_PROGRESS_SECONDS):
        self.enabled = enabled
        self.log_seconds = log_seconds
        self.duration = 0.0
        self.started = time.monotonic()
        self._last_line = 0.0
        self._last_percent = -1.0
        self._last_milestone = -1
        self._tty = sys.stderr.isatty()

    def _percent(self, processed_seconds: float) -> float:
        if self.duration <= 0:
            return 0.0
        return min(max(processed_seconds / self.duration * 100.0, 0.0), 100.0)

    def _message(self, processed_seconds: float, percent: float, now: float) -> str:
        elapsed = max(now - self.started, 1e-6)
        parts = [f"progress {percent:5.1f}%"] if self.duration > 0 else ["decoding"]
        if self.duration > 0:
            parts.append(f"{clock(processed_seconds)}/{clock(self.duration)}")
        else:
            parts.append(f"{clock(processed_seconds)} decoded")
        parts.append(f"elapsed {clock(elapsed)}")
        if 1.0 <= percent < 100.0:
            parts.append(f"eta {clock(elapsed * (100.0 - percent) / percent)}")
        if processed_seconds > 0:
            parts.append(f"{processed_seconds / elapsed:.2f}x")
        return "  ".join(parts)

    def _emit(self, message: str, persistent: bool) -> None:
        self._last_line = time.monotonic()
        if persistent:
            if self._tty:
                print("\r" + " " * 78 + "\r", end="", file=sys.stderr, flush=True)
            print(message, file=sys.stderr, flush=True)
        else:
            print(f"\r{message:<78}", end="", file=sys.stderr, flush=True)

    def update(self, processed_seconds: float) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        percent = self._percent(processed_seconds)
        self._last_percent = percent

        # Every 10% becomes a permanent line, so even a redirected log keeps a
        # compact trail of how far the decode has come.
        milestone = int(percent // 10)
        if milestone > self._last_milestone:
            self._last_milestone = milestone
            self._emit(self._message(processed_seconds, percent, now), persistent=True)
            return

        if self._tty:
            if now - self._last_line < 1.0:
                return
        elif self.log_seconds <= 0 or now - self._last_line < self.log_seconds:
            return
        self._emit(self._message(processed_seconds, percent, now), persistent=not self._tty)

    def finish(self, processed_seconds: float | None = None) -> None:
        """Log the closing percentage and clear the live line on a terminal."""
        if not self.enabled:
            return
        if processed_seconds is not None and self._last_percent < 99.9:
            percent = self._percent(processed_seconds)
            self._emit(self._message(processed_seconds, percent, time.monotonic()), persistent=True)
        if self._tty:
            print("\r" + " " * 78 + "\r", end="", file=sys.stderr, flush=True)


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to ``cuda`` when CTranslate2 reports a usable GPU."""
    if requested != "auto":
        return requested
    try:
        import ctranslate2
    except Exception:  # pragma: no cover - dependency missing
        return "cpu"
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # pragma: no cover - driver/device query failure
        return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested:
        return requested
    return "float16" if device == "cuda" else "int8"


def model_key(model_name: str) -> str:
    """Folder name a model gets inside the local model directory.

    ``distil-large-v3`` stays ``distil-large-v3``; a Hub repo id such as
    ``Systran/faster-whisper-large-v3`` becomes ``Systran__faster-whisper-large-v3``.
    """
    return sanitize_stem(model_name.replace("/", "__"))


def local_model_dir(model_name: str, model_dir: str | Path) -> Path:
    """``<model_dir>/<model_key>``: the folder that holds one downloaded model."""
    return Path(model_dir).expanduser() / model_key(model_name)


def ensure_model(model_name: str, model_dir: str | Path, verbose: bool = True) -> str:
    """Return a model reference that is guaranteed to exist on this machine.

    A local CTranslate2 directory is passed through untouched. A model name (or
    Hub repo id) is looked up in ``model_dir`` first; when it is not there yet the
    weights are downloaded *now*, before any audio is decoded, so a missing model
    or a dead network fails immediately instead of halfway through a long file.
    HuggingFace keeps its own cache, so a model that already sits in
    ``~/.cache/huggingface`` is copied instead of downloaded a second time.
    """
    local = Path(model_name).expanduser()
    if local.is_dir():
        if (local / MODEL_BIN).exists():
            return str(local)
        raise SystemExit(
            f"error: {local} does not look like a CTranslate2 model directory "
            f"(no {MODEL_BIN} inside)\n"
            "hint: use a known model name (--list-models) or a directory this script downloaded"
        )
    if model_name.endswith(".bin") or model_name.startswith((".", "/", "~")):
        raise SystemExit(
            f"error: no such model directory: {model_name}\n"
            "hint: use a known model name (--list-models) or an existing local CTranslate2 directory"
        )

    if "/" not in model_name and model_name not in KNOWN_MODELS:
        # Catch a typo here, before a folder is created and a download is attempted.
        known = ", ".join(KNOWN_MODELS)
        raise SystemExit(
            f"error: unknown model {model_name!r}\n"
            f"hint: use a known name ({known}) or a local CTranslate2 directory"
        )

    target = local_model_dir(model_name, model_dir)
    if (target / MODEL_BIN).exists():
        if verbose:
            print(f"Model       : {model_name} (found in {target})")
        return str(target)

    _require_faster_whisper()
    from faster_whisper.utils import download_model

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"error: cannot create model directory {target.parent}: {exc}") from exc

    print(f"Model       : {model_name} is not in {target.parent} yet")
    print(f"              downloading into {target} (one time, later runs reuse it) ...", flush=True)
    try:
        with warnings.catch_warnings():
            # faster-whisper 1.2.1 still passes `local_dir_use_symlinks`, which the
            # installed huggingface_hub ignores; that notice is noise for the user.
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
            path = download_model(model_name, output_dir=str(target))
    except ValueError as exc:
        known = ", ".join(KNOWN_MODELS)
        raise SystemExit(
            f"error: could not download model {model_name!r}: {exc}\n"
            f"hint: use a known name ({known}) or a local CTranslate2 directory"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"error: could not download model {model_name!r} into {target}: {exc}\n"
            "hint: check your network connection, use --hf-cache to reuse the shared "
            "HuggingFace cache, or point --model at a local CTranslate2 directory"
        ) from exc

    if verbose:
        print(f"              saved to {path}")
    return path


def _require_faster_whisper():
    """Import faster-whisper, or exit with an actionable install hint."""
    try:
        import faster_whisper
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "error: faster-whisper is not installed.\n"
            "hint: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        ) from exc
    return faster_whisper


def load_model(model_name: str, device: str, compute_type: str, verbose: bool):
    """Instantiate the Whisper model (``ensure_model`` already made sure it exists)."""
    from faster_whisper import WhisperModel

    if verbose:
        print(f"Loading model {model_name!r} (device={device}, compute_type={compute_type}) ...")

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except ValueError as exc:
        known = ", ".join(KNOWN_MODELS)
        raise SystemExit(
            f"error: could not load model {model_name!r}: {exc}\n"
            f"hint: use a known name ({known}) or a local CTranslate2 directory"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"error: could not load model {model_name!r}: {exc}\n"
            "hint: check your network connection, or point --model at a local CTranslate2 directory"
        ) from exc

    if verbose:
        try:
            multilingual = len(model.supported_languages) > 1
        except Exception:  # pragma: no cover - defensive
            multilingual = not is_english_only(model_name)
        print(f"Model ready (multilingual={multilingual}).")
    return model


def file_output_dir(path: Path, output_dir: Path, flat: bool = False) -> Path:
    """Folder that holds one input's transcripts.

    One sub-folder per input keeps every artifact of a recording together
    (``transcripts/<stem>/``); ``flat=True`` (``--flat``) writes directly into the
    output directory instead.
    """
    if flat:
        return output_dir
    return output_dir / sanitize_stem(path.stem)


def expected_outputs(path: Path, output_dir: Path, formats, flat: bool = False) -> list[Path]:
    """The output file paths a given input would produce."""
    folder = file_output_dir(path, output_dir, flat)
    stem = sanitize_stem(path.stem)
    return [folder / f"{stem}.{fmt}" for fmt in formats]


def partial_outputs(path: Path, output_dir: Path, formats, flat: bool = False) -> list[Path]:
    """Checkpoint paths written while a long file is still being decoded."""
    folder = file_output_dir(path, output_dir, flat)
    stem = sanitize_stem(path.stem)
    return [folder / f"{stem}.partial.{fmt}" for fmt in formats]


def write_outputs(
    path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    compute_type: str,
    info,
    segments: list[Segment],
    verbose: bool,
    partial: bool = False,
) -> list[Path]:
    """Render and write every requested format into the input's own folder.

    ``partial=True`` writes ``<stem>.partial.<ext>`` checkpoints while a long file
    is still decoding; the final call writes the real names and removes the
    checkpoints, so a half-finished file can never masquerade as a finished one.
    """
    stem = sanitize_stem(path.stem)
    folder = file_output_dir(path, output_dir, args.flat)
    folder.mkdir(parents=True, exist_ok=True)
    rendered = {
        "txt": lambda: render_txt(segments, with_timestamps=args.timestamps),
        "srt": lambda: render_srt(segments),
        "vtt": lambda: render_vtt(segments),
        "json": lambda: render_json(build_json_payload(path, args, device, compute_type, info, segments)),
    }

    written = []
    for fmt in args.formats:
        target = folder / (f"{stem}.partial.{fmt}" if partial else f"{stem}.{fmt}")
        if not partial and args.skip_existing and target.exists():
            if verbose:
                print(f"  skip (already exists): {target}")
            continue
        target.write_text(rendered[fmt](), encoding="utf-8")
        written.append(target)
        if verbose:
            print(f"  wrote {target}")

    if not partial:
        for leftover in partial_outputs(path, output_dir, args.formats, args.flat):
            leftover.unlink(missing_ok=True)
    return written


def transcribe_file(
    model,
    path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    compute_type: str,
    verbose: bool,
) -> tuple[list[Path], float, int, float]:
    """Transcribe one media file, print progress, and write its outputs.

    Returns ``(written_paths, elapsed_seconds, segment_count, audio_seconds)``.
    """
    if verbose:
        print(f"\n=== {path.name} ===")

    reporter = ProgressReporter(verbose and not args.log_progress, args.progress_seconds)
    transcribe_kwargs = {
        "beam_size": args.beam_size,
        "task": args.task,
        "vad_filter": not args.no_vad,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
        "log_progress": args.log_progress,
    }
    if args.language:
        transcribe_kwargs["language"] = args.language
    if args.initial_prompt:
        transcribe_kwargs["initial_prompt"] = args.initial_prompt
    if args.hotwords and args.hotwords.strip():
        # faster-whisper 1.2.x expects a single string and tokenizes it itself.
        transcribe_kwargs["hotwords"] = args.hotwords.strip()
    if not args.no_vad:
        vad_parameters = {"min_silence_duration_ms": args.vad_min_silence_ms}
        if args.vad_threshold is not None:
            vad_parameters["threshold"] = args.vad_threshold
        transcribe_kwargs["vad_parameters"] = vad_parameters

    started = time.monotonic()
    # transcribe() returns a lazy generator, so `info` is available before the
    # first segment is decoded - that keeps the progress bar's total accurate.
    raw_segments, info = model.transcribe(str(path), **transcribe_kwargs)
    reporter.duration = float(getattr(info, "duration", 0.0) or 0.0)

    segments: list[Segment] = []
    last_checkpoint = time.monotonic()
    try:
        for raw_segment in raw_segments:
            segment = to_segment(raw_segment, args.word_timestamps)
            segments.append(segment)
            reporter.update(segment.end)
            if verbose and segment.text:
                print(f"[{clock(segment.start)} -> {clock(segment.end)}] {segment.text}")

            # Checkpoint long files so a crash or Ctrl-C never loses an hour of work.
            if args.checkpoint_seconds and time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                last_checkpoint = time.monotonic()
                write_outputs(
                    path, output_dir, args, device, compute_type, info, segments, verbose, partial=True
                )
                if verbose:
                    print(f"  checkpoint: {len(segments)} segments decoded up to {clock(segment.end)}")
    except KeyboardInterrupt:
        reporter.finish(segments[-1].end if segments else None)
        rescued = write_outputs(
            path, output_dir, args, device, compute_type, info, segments, verbose, partial=True
        )
        print(
            f"\ninterrupted after {len(segments)} segment(s) ({clock(segments[-1].end) if segments else '00:00'} "
            "of audio); partial transcript kept in "
            + ", ".join(str(target) for target in rescued),
            file=sys.stderr,
        )
        raise

    reporter.finish(segments[-1].end if segments else None)
    elapsed = time.monotonic() - started

    if verbose:
        print(
            f"Language: {info.language} "
            f"(p={info.language_probability:.2f}), "
            f"audio {human_duration(info.duration)}, "
            f"{len(segments)} segments, decoded in {human_duration(elapsed)}"
            + (f" ({info.duration / elapsed:.2f}x realtime)" if info.duration else "")
        )

    if not any(segment.text for segment in segments):
        print(f"warning: no speech detected in {path.name} (empty transcript written)", file=sys.stderr)

    written = write_outputs(path, output_dir, args, device, compute_type, info, segments, verbose)
    return written, elapsed, len(segments), float(getattr(info, "duration", 0.0) or 0.0)


MODEL_TABLE = """\
  model              38 min of audio (14-core CPU, int8)    notes
  tiny                 1 min (38x realtime, measured)       rough, smoke tests only
  base                ~2 min                                okay
  small               ~4 min                                good quality/speed balance
  distil-large-v3      8m38s (4.4x realtime, measured)      DEFAULT, near large-v3, English only
  medium             ~13 min                                solid, multilingual
  large-v3           ~50 min                                best accuracy, multilingual
"""

EPILOG = f"""\
model size vs speed (int8 on CPU, ~38 min of audio):
{MODEL_TABLE}
examples:
  python transcribe.py ytaudio/                            # transcribe a whole folder
  python transcribe.py talk.mp3 -f txt,srt,json            # all output formats
  python transcribe.py lecture.wav -m large-v3 -l de       # German, best accuracy
  python transcribe.py podcast.mp3 --initial-prompt "Andrew Ng, AI Fund"
  python transcribe.py ytaudio/ --skip-existing            # resumable batch
  python transcribe.py ytaudio/ --list-files               # numbered files to pick from
  python transcribe.py ytaudio/ --only "Andrew Ng"         # transcribe a single recording
  python transcribe.py --prefetch-model                    # download the model now, transcribe nothing
"""


def expand_inputs(raw_inputs: list[str]) -> list[Path]:
    """Resolve CLI arguments into a sorted, de-duplicated list of media files."""
    found: list[Path] = []
    seen: set[Path] = set()

    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in MEDIA_EXTENSIONS
            )
            if not candidates:
                print(f"warning: no media files found under {path}", file=sys.stderr)
        elif path.is_file():
            candidates = [path]
        else:
            print(f"error: no such file or directory: {path}", file=sys.stderr)
            continue

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)

    return found


def matches_name(name: str, pattern: str) -> bool:
    """Case-insensitive substring match, or a glob when the pattern uses wildcards."""
    lowered = pattern.lower()
    if any(char in lowered for char in "*?["):
        return fnmatch.fnmatchcase(name.lower(), lowered)
    return lowered in name.lower()


def select_files(files: list[Path], only: list[str], exclude: list[str]) -> list[Path]:
    """Apply the ``--only`` / ``--exclude`` name filters to the discovered files."""
    selected = []
    for path in files:
        if only and not any(matches_name(path.name, pattern) for pattern in only):
            continue
        if exclude and any(matches_name(path.name, pattern) for pattern in exclude):
            continue
        selected.append(path)
    return selected


def parse_picks(raw_values: list[str], parser: argparse.ArgumentParser) -> list[int]:
    """Turn ``--pick`` values (``3``, ``1,4``, ``2-5``) into sorted 1-based indices."""
    picks: set[int] = set()
    for raw in raw_values:
        for part in (value for value in re.split(r"[,\s]+", raw.strip()) if value):
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
            if not match:
                parser.error(f"--pick expects a number or range such as 3 or 2-5, got {part!r}")
            first = int(match.group(1))
            last = int(match.group(2)) if match.group(2) else first
            if first < 1 or last < first:
                parser.error(f"--pick {part!r} is not a valid 1-based range")
            picks.update(range(first, last + 1))
    return sorted(picks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe.py",
        description=(
            "Transcribe audio/video files (mp3, flac, wav, m4a, ogg, opus, mp4, ...) to text "
            "using faster-whisper (Whisper on the CTranslate2 runtime)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="AUDIO_OR_DIR",
        help="audio/video file(s); a directory is scanned recursively for media files",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "transcribe only files whose name contains TEXT (case-insensitive; wildcards "
            "like '*lecture*' work; repeat for more names)"
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TEXT",
        help="skip files whose name contains TEXT (case-insensitive; repeatable)",
    )
    parser.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="N",
        help=(
            "transcribe only the Nth selected file(s), numbered like --list-files "
            "(e.g. 3, or 1,4, or 2-5)"
        ),
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="print the files that would be transcribed, with their --pick numbers, then exit",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"where transcripts are written (default: ./{DEFAULT_OUTPUT_DIR}/)",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name or local CTranslate2 directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="spoken language code, e.g. en/de/es; omit to auto-detect",
    )
    parser.add_argument(
        "-f", "--format",
        dest="format",
        default=",".join(DEFAULT_FORMATS),
        help=f"comma-separated output formats from {list(ALL_FORMATS)} (default: {','.join(DEFAULT_FORMATS)})",
    )
    parser.add_argument(
        "-d", "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="inference device (default: auto -> cuda when available, else cpu)",
    )
    parser.add_argument(
        "-c", "--compute-type",
        default="",
        help="CTranslate2 compute type, e.g. int8/int8_float32/float32/float16 (default: int8 on cpu, float16 on cuda)",
    )
    parser.add_argument(
        "-t", "--task",
        default="transcribe",
        choices=("transcribe", "translate"),
        help="transcribe in the source language, or translate the audio into English (default: transcribe)",
    )
    parser.add_argument(
        "-b", "--beam-size",
        type=int,
        default=5,
        help="beam search width; 1 is greedy and fastest (default: 5)",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="text prompt that biases decoding, useful for names and jargon",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="comma-separated words/phrases to boost during decoding",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=(
            "folder that holds the downloaded models "
            f"(default: {DEFAULT_MODEL_DIR}; created on first use, gitignored)"
        ),
    )
    parser.add_argument(
        "--hf-cache",
        action="store_true",
        help=(
            "keep models in the shared HuggingFace cache (~/.cache/huggingface) "
            "instead of the repo-local --model-dir"
        ),
    )
    parser.add_argument(
        "--prefetch-model",
        action="store_true",
        help="download the selected model into --model-dir if it is missing, then exit",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="also produce word-level timestamps (slower; adds a 'words' list to --format json)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="prefix each line of the .txt transcript with its start time",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="disable the Silero VAD silence filter (leave it on unless you need every second)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="VAD speech probability threshold, e.g. 0.5 (default: 0.5)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="minimum silence length in ms that splits speech chunks (default: 500)",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="decode each window fresh; can stop repetition loops on noisy audio",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=int,
        default=DEFAULT_CHECKPOINT_SECONDS,
        help=(
            "while decoding, save a '<name>.partial.<ext>' transcript every N seconds so an "
            f"interrupted run keeps its work; 0 disables (default: {DEFAULT_CHECKPOINT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip files whose output already exists, making long batches resumable",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the model table and exit",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="use the library's tqdm progress bar instead of the built-in progress line",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=DEFAULT_PROGRESS_SECONDS,
        help=(
            "when stderr is not a terminal, log a progress line at least every N seconds; "
            "one line per 10%% is always logged (0 = only those; "
            f"default: {DEFAULT_PROGRESS_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help=(
            "write every transcript straight into the output directory instead of "
            "one sub-folder per input file"
        ),
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="only print the final summary and any errors",
    )
    return parser


def parse_formats(raw: str, parser: argparse.ArgumentParser) -> list[str]:
    """Validate ``--format txt,srt`` into an ordered list of unique formats."""
    formats: list[str] = []
    for part in raw.split(","):
        fmt = part.strip().lower()
        if not fmt:
            continue
        if fmt not in ALL_FORMATS:
            parser.error(f"unknown format {fmt!r}; choose from: {', '.join(ALL_FORMATS)}")
        if fmt not in formats:
            formats.append(fmt)
    if not formats:
        parser.error(f"--format needs at least one of: {', '.join(ALL_FORMATS)}")
    return formats


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print("known Whisper models:")
        print(MODEL_TABLE)
        return 0

    args.formats = parse_formats(args.format, parser)
    verbose = not args.quiet

    if args.prefetch_model:
        # Fetch the weights and stop, so a fresh clone or a container image can be
        # warmed up and the first real transcription never waits on the download.
        if args.hf_cache:
            print(
                "error: --prefetch-model downloads into --model-dir; drop --hf-cache",
                file=sys.stderr,
            )
            return 2
        path = ensure_model(args.model, args.model_dir, verbose=verbose)
        print(f"model ready: {path}")
        return 0

    if not args.inputs:
        parser.error("at least one AUDIO_OR_DIR is required (use --list-models to see available models)")

    if args.language:
        args.language = args.language.strip().lower()
        if args.language in ("auto", "auto-detect", "detect"):
            args.language = None

    # English-only checkpoints (distil-*, *.en) cannot translate, and must not be
    # fed non-English audio. Pinning the language also skips a useless detection pass.
    if is_english_only(args.model):
        if args.task == "translate":
            parser.error(
                f"model {args.model!r} is English-only and cannot translate; "
                "use --model large-v3 when translating"
            )
        if args.language and args.language != "en":
            parser.error(
                f"model {args.model!r} only understands English, but --language {args.language!r} "
                "was requested; use --model large-v3 (or medium) for other languages"
            )
        if not args.language:
            args.language = "en"
            if verbose and not args.list_files:
                print(f"English-only model {args.model!r}: forcing language=en (no auto-detect).")

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)

    discovered = expand_inputs(args.inputs)
    if not discovered:
        print("error: nothing to transcribe", file=sys.stderr)
        return 2

    # Narrow the discovered files down to what the user actually asked for, so a
    # single recording can be picked out of a folder without retyping a long name.
    files = select_files(discovered, args.only, args.exclude)
    picks = parse_picks(args.pick, parser) if args.pick else []
    if picks and picks[-1] > len(files):
        parser.error(
            f"--pick {picks[-1]} is out of range: only {len(files)} file(s) selected "
            "(use --list-files to see the numbers)"
        )

    if args.list_files:
        # Numbers stay those of the --only/--exclude result, so a later --pick N
        # refers to exactly what was printed here.
        chosen = [path for index, path in enumerate(files, start=1) if index in picks]
        for index, path in enumerate(files, start=1):
            print(f"{index:3d}. {path.name}{'  *' if picks and index in picks else ''}")
        print(
            f"{len(chosen) if picks else len(files)} of {len(discovered)} discovered "
            "file(s) would be transcribed"
            + ("  (* = selected by --pick)" if picks else "")
        )
        return 0

    if picks:
        files = [path for index, path in enumerate(files, start=1) if index in picks]

    if not files:
        print(
            "error: no file selected -- --only/--exclude/--pick filtered out every input",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir).expanduser()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create output directory {output_dir}: {exc}", file=sys.stderr)
        return 2

    if verbose:
        if len(files) != len(discovered):
            print(
                f"Input files : {len(files)} of {len(discovered)} "
                "(selected by --only/--exclude/--pick)"
            )
        else:
            print(f"Input files : {len(files)}")
        print(f"Output dir  : {output_dir.resolve()}")
        print(f"Layout      : {'flat (all files side by side)' if args.flat else 'one sub-folder per input'}")
        print(f"Formats     : {', '.join(args.formats)}")

    if args.hf_cache:
        # Old behaviour: let the library manage the shared HuggingFace cache.
        model_ref = args.model
        if verbose:
            print(f"Model       : {args.model} (shared HuggingFace cache)")
    else:
        # Resolve (and, when absent, download) the weights before any audio is
        # touched, so a missing model or a dead network fails immediately.
        model_ref = ensure_model(args.model, args.model_dir, verbose)

    model = load_model(model_ref, device, compute_type, verbose)

    written_all: list[Path] = []
    failures: list[tuple[Path, str]] = []
    skipped = 0
    total_audio = 0.0
    total_elapsed = 0.0

    for index, path in enumerate(files, start=1):
        if verbose and len(files) > 1:
            print(f"\n[{index}/{len(files)}] {path.name}")

        # Skip the expensive decoding entirely when every requested output exists.
        if args.skip_existing and all(
            target.exists() for target in expected_outputs(path, output_dir, args.formats, args.flat)
        ):
            skipped += 1
            if verbose:
                print("skip (all requested outputs already exist)")
            continue

        try:
            written, elapsed, _segments, audio_duration = transcribe_file(
                model, path, output_dir, args, device, compute_type, verbose
            )
        except KeyboardInterrupt:
            print("\ninterrupted by user", file=sys.stderr)
            return 130
        except Exception as exc:
            message = str(exc)
            if "metadata_errors" in message:
                message += (
                    "\n  hint: PyAV 19+ dropped a kwarg that faster-whisper 1.2.1 relies on; "
                    "fix the pinned dependencies with: .venv/bin/pip install -r requirements.txt"
                )
            print(f"error: failed to transcribe {path}: {message}", file=sys.stderr)
            failures.append((path, message))
            continue
        written_all.extend(written)
        total_elapsed += elapsed
        total_audio += audio_duration

    if verbose and written_all:
        print("\n" + "-" * 60)
        for target in written_all:
            print(f"  {target}")

    speedup = f"{total_audio / total_elapsed:.2f}x realtime" if total_elapsed > 0 else "n/a"
    summary = (
        f"Done: {len(files) - len(failures) - skipped}/{len(files)} file(s) transcribed, "
        f"{len(written_all)} output file(s) in {output_dir.resolve()}"
    )
    if skipped:
        summary += f", {skipped} file(s) skipped (outputs already existed)"
    if total_audio:
        summary += (
            f" (decoded {human_duration(total_audio)} of audio in "
            f"{human_duration(total_elapsed)}, {speedup})"
        )
    print(summary)

    if failures:
        for path, message in failures:
            print(f"  failed: {path.name}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
