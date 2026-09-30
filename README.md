# ytsummarizer

Turn YouTube/podcast audio into text, subtitles and structured JSON — locally,
with no cloud API and no API keys.

`transcribe.py` accepts any audio or video file (mp3, flac, wav, m4a, ogg, opus,
aac, mp4, mkv, webm, ...) or a whole folder, and writes each transcript into its
own folder (`./transcripts/<file name>/` by default). Pass one file path to
transcribe just that recording, or filter a folder by name or number — see
[Pick a single file](#pick-a-single-file).

## Why faster-whisper

| Reason | Detail |
|---|---|
| Fast on CPU | CTranslate2 runtime with `int8` quantization: ~4x faster than `openai-whisper`, and no ~2.5 GB torch install |
| Same accuracy | Runs the official Whisper checkpoints, not a reimplementation |
| Any format, no ffmpeg step | Decodes through PyAV, so wav/flac/mp3/m4a/ogg/opus and even video containers work directly |
| Subtitles for free | Segment timestamps drive `.srt`/`.vtt`; optional word-level timestamps land in `.json` |
| Silence handling | Bundled Silero VAD skips gaps — faster, and no hallucinated text in silent stretches |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The model weights are downloaded into the checkout, under `./models/<model-name>/`
(gitignored — only the script and docs are tracked). The first run for a model
fetches it (~1.5 GB for `distil-large-v3`, ~75 MB for `tiny`) and prints where it
landed; later runs find the folder and need no network. A model that already lives
in `~/.cache/huggingface` is copied instead of downloaded again. To warm that up
ahead of time — handy in CI or Docker — use `--prefetch-model`:

```bash
.venv/bin/python transcribe.py --prefetch-model            # default: distil-large-v3
.venv/bin/python transcribe.py --prefetch-model -m tiny    # ~75 MB smoke test
```

Use `--hf-cache` to keep models in the shared `~/.cache/huggingface` instead, or
`--model-dir PATH` to put them somewhere else.

## Run

```bash
# whole folder (recursive) with the defaults: distil-large-v3, txt + srt, ./transcripts/
.venv/bin/python transcribe.py ytaudio/

# one file plus every output format
.venv/bin/python transcribe.py ytaudio/*.wav -f txt,srt,vtt,json

# other languages (distil-* and *.en models are English-only)
.venv/bin/python transcribe.py podcast.mp3 -m large-v3 -l de

# bias decoding toward names/jargon, and make a long batch resumable
.venv/bin/python transcribe.py talk.mp3 --initial-prompt "Andrew Ng, AI Fund"
.venv/bin/python transcribe.py ytaudio/ --skip-existing

# checkpoint a very long file more often; Ctrl-C keeps <name>.partial.<ext>
.venv/bin/python transcribe.py lecture.wav --checkpoint-seconds 30
```

While a file decodes you get a live percentage line on a terminal — and in a
redirected log (`nohup`, `tee`, `> run.log`) one line at every 10% plus a
heartbeat every 15 s (`--progress-seconds`) — followed by a summary with the
written paths and the realtime factor. `-q` keeps only that summary:

```text
progress  42.4%  16:03/37:51  elapsed 03:52  eta 05:07  4.19x
```

## Pick a single file

A directory argument means "everything inside it". To narrow that down without
retyping a long name (spaces, apostrophes and full-width punctuation are no fun
in a shell), filter by name or by number:

```bash
# 1. … 2. … — numbered list of what would be transcribed, then exit
.venv/bin/python transcribe.py ytaudio/ --list-files

# keep only files whose name contains TEXT (case-insensitive; '*lecture*' globs work; repeatable)
.venv/bin/python transcribe.py ytaudio/ --only "Andrew Ng"

# keep the Nth file(s) from --list-files: 1, '1,4' and '2-5' all work
.venv/bin/python transcribe.py ytaudio/ --pick 1

# keep everything except matches (repeatable)
.venv/bin/python transcribe.py ytaudio/ --exclude trailer
```

`--only` runs first, then `--exclude`, then `--pick` numbers the survivors, so the
numbers always match the `--list-files` output. An explicit path keeps working
too: `.venv/bin/python transcribe.py 'ytaudio/My Talk： Part 1.wav'`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-o, --output-dir` | `./transcripts` | Directory that receives the transcripts (created if missing) |
| `-m, --model` | `distil-large-v3` | Whisper model name, or a local CTranslate2 directory |
| `-l, --language` | auto-detect | Language code such as `en`, `de`, `es` |
| `-f, --format` | `txt,srt` | Any combination of `txt`, `srt`, `vtt`, `json` |
| `--only` | – | Transcribe only files whose name contains `TEXT` (case-insensitive, globs allowed; repeatable) |
| `--exclude` | – | Skip files whose name contains `TEXT` (case-insensitive; repeatable) |
| `--pick` | – | Transcribe only the Nth file(s) as numbered by `--list-files` (`3`, `1,4`, `2-5`) |
| `--list-files` | – | Print the files that would be transcribed, with their `--pick` numbers, then exit |
| `-d, --device` | `auto` | `auto` (cuda when available, else cpu), `cpu`, `cuda` |
| `-c, --compute-type` | `int8` on cpu, `float16` on cuda | CTranslate2 compute type (`int8`, `int8_float32`, `float32`, ...) |
| `-t, --task` | `transcribe` | `translate` renders any language into English (needs a multilingual model) |
| `-b, --beam-size` | `5` | Beam search width; `1` is greedy and fastest |
| `--initial-prompt` | – | Prompt text that biases spelling of names and jargon |
| `--hotwords` | – | Comma-separated terms to boost during decoding |
| `--word-timestamps` | off | Adds a `words` list to the `.json` output (slower) |
| `--timestamps` | off | Prefixes every `.txt` line with its start time |
| `--no-vad` | off | Disable the Silero VAD silence filter |
| `--vad-threshold` | `0.5` | VAD speech-probability threshold |
| `--vad-min-silence-ms` | `500` | Silence length that splits speech chunks |
| `--no-condition-on-previous-text` | off | Decode each window fresh; can stop repetition loops |
| `--model-dir` | `./models` | Repo-local folder that holds downloaded models (gitignored; created on first use) |
| `--hf-cache` | off | Keep models in the shared `~/.cache/huggingface` instead of `--model-dir` |
| `--prefetch-model` | – | Download the selected model into `--model-dir` if missing, then exit |
| `--checkpoint-seconds` | `60` | Flush a `<name>.partial.<ext>` transcript every N seconds while decoding, so an interrupted run keeps its work (`0` disables) |
| `--skip-existing` | off | Skip files whose outputs all exist — decoding included, not just the write (resumable batches) |
| `--log-progress` | off | Show the library's own tqdm bar instead of the built-in progress line |
| `--progress-seconds` | `15` | When stderr is not a terminal, log a progress line at least every N seconds (a line per 10% is always logged; `0` = only those) |
| `--flat` | off | Write every transcript straight into the output directory instead of one sub-folder per input |
| `-q, --quiet` | off | Only print the final summary and errors |
| `--list-models` | – | Print the model size/speed table and exit |

Run `.venv/bin/python transcribe.py --help` for the full reference.

## Output

Every input gets its own folder named after its sanitized stem, so all artifacts
of one recording stay together — including notes or summaries you write yourself:

```
transcripts/
  Andrew Ng： The Biggest Opportunities in AI Aren't Where You Think/
    Andrew Ng： The Biggest Opportunities in AI Aren't Where You Think.txt
    Andrew Ng： The Biggest Opportunities in AI Aren't Where You Think.srt
    Andrew Ng： The Biggest Opportunities in AI Aren't Where You Think.json
    Andrew Ng： The Biggest Opportunities in AI Aren't Where You Think.md   ← your own summary
```

`--flat` skips the per-file folder and writes straight into the output directory
(the old layout: every file in one big pile).

- **`.txt`** — plain transcript, one line per segment (`--timestamps` prefixes `[mm:ss]`).
- **`.srt`** — SubRip subtitles with `HH:MM:SS,mmm` timings, ready for any player or editor.
- **`.vtt`** — WebVTT for web players and `<track>` elements.
- **`.json`** — machine readable: source file, model, device, compute type, task,
  detected language + probability, duration, segment count, generation timestamp,
  the joined `transcript` string, and every timed segment with `start`/`end`/`text`,
  plus `avg_logprob`/`no_speech_prob` and (with `--word-timestamps`) a `words` list.

The input audio is never modified, and only files that are actually requested are
written.

## Models

| Model | 38 min of audio, 14-core CPU (int8) | Notes |
|---|---|---|
| `tiny` | 1 min — 38x realtime *(measured)* | rough, good for smoke tests |
| `base` | ~2 min | okay |
| `small` | ~4 min | good quality/speed balance |
| `distil-large-v3` | 8m38s — 4.4x realtime *(measured)* | **default** — near large-v3 accuracy, English only |
| `medium` | ~13 min | solid, multilingual |
| `large-v3` | ~50 min | best accuracy, multilingual |

The two measured rows come from a real 37m51s podcast (`distil-large-v3`:
519 segments, 7,169 words, `language_probability=1.0`) and a full pass with
`tiny` over the same file; the other rows are scaled estimates.

`distil-*` and `*.en` checkpoints only understand English. For them the script
refuses `--task translate` and non-English `--language` values with a clear
message pointing at `large-v3`, and it pins `language=en` so no detection pass is
wasted.

## Notes and caveats

- **Pinned PyAV.** faster-whisper 1.2.1 calls `av.open(..., metadata_errors="ignore")`,
  a kwarg removed in PyAV 19.0.0, so `requirements.txt` pins `av>=14,<19`. If you
  ever see `open() got an unexpected keyword argument 'metadata_errors'`, re-run
  `.venv/bin/pip install -r requirements.txt` (the CLI also prints this hint).
- **Resources.** int8 on CPU keeps a 38-minute 48 kHz stereo WAV inside a few GB of
  RAM; the model folder costs 75 MB (`tiny`) up to ~3 GB (`large-v3`).
- **Models live in the repo.** `./models/<model-name>/` holds the CTranslate2
  weights and is gitignored, so a fresh clone downloads them on first use
  (`--prefetch-model` does it ahead of time). `--hf-cache` reuses the shared
  `~/.cache/huggingface` cache instead, and `--model-dir` relocates the folder.
- **Long files are checkpointed.** While decoding, the transcript so far is
  flushed to `<name>.partial.<ext>` — inside the file's own folder — every 60 s
  (`--checkpoint-seconds`). If you
  hit `Ctrl-C` mid-file, that checkpoint is kept and the CLI tells you where it
  is; the real `<name>.<ext>` files are only written once the file is finished,
  and the checkpoints are then deleted. A `.partial.*` file is never treated as
  a finished transcript, so `--skip-existing` still redoes an interrupted file.
- **Resumable batches.** Each file is handled independently; with
  `--skip-existing`, files whose outputs already exist are skipped before any
  decoding starts, so re-running a folder costs seconds, not hours.
- **VAD is on by default.** It removes long silences and the hallucinated text
  Whisper tends to invent there. Use `--no-vad` when you need a verbatim timeline.
- **Repetition loops?** On noisy audio, `--no-condition-on-previous-text` often
  breaks a repeating phrase.
