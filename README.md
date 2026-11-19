# Short Captioner

Short Captioner is a small, local video-captioning app for macOS. Select a video in your browser, choose a caption style and layout, and create a captioned MP4. The app also creates an SRT subtitle file that you can download separately.

Speech recognition and video rendering run on your computer. The project does not send your video or transcript to an AI service.

## What you need

- macOS
- Python 3.10 or newer (Python 3.12 is recommended)
- FFmpeg with the `ass` subtitle filter, plus `ffprobe`
- A modern web browser
- About 480 MB of free space for the speech model if it is not already available from the neighboring PeakCut project
- Enough free disk space for your uploaded source video and rendered output

The launcher installs the Python packages listed in `requirements.txt` into a project-local `.venv` the first time it runs. The app accepts MP4, MOV, M4V, MKV, WebM, and AVI files. The default upload limit is 20 GB.

## Install and run on macOS

### 1. Install Python and FFmpeg

With Homebrew:

```sh
brew install python@3.12 ffmpeg-full
```

The full FFmpeg build is needed because caption rendering uses FFmpeg's `ass` filter. `ffprobe` is used to read the video's dimensions, duration, and audio/video streams.

### 2. Start the app

In Terminal, move into this project folder and run the launcher:

```sh
cd /path/to/local-captioner
./run.sh
```

On its first run, the launcher creates `.venv` and installs the Python requirements. Leave this Terminal window open while you use the app.

### 3. Open Short Captioner

Visit [http://127.0.0.1:8766](http://127.0.0.1:8766) in your browser. The app listens on `127.0.0.1`, so it is available on this computer only by default.

To stop the server, return to Terminal and press **Ctrl+C**.

## First run and the speech model

The app uses the local **Parakeet TDT 0.6B int8** speech-recognition model through Sherpa-ONNX. It needs these model files: `tokens.txt`, `encoder.int8.onnx`, `decoder.int8.onnx`, and `joiner.int8.onnx`.

Short Captioner checks for a compatible model in this order:

1. The folder set in `CAPTION_MODEL_DIR`, if that environment variable is set.
2. A model already stored in the neighboring `../peakcut/data/models/` folder.
3. This project's `data/models/` folder.

If no compatible model is found, click **Download model** in the app. It downloads the model once from the Sherpa-ONNX project and stores it locally. The initial setup needs an internet connection; transcription and rendering afterward run locally.

## Create captions

1. Drop a video onto the upload area or click **Choose video**.
2. Choose a caption style and settings.
3. Click **Create captions** and wait for the job to finish.
4. Download the captioned MP4, and download the SRT separately if you need it.

The app keeps completed videos in the **Your videos** list. Longer videos take longer to transcribe and render.

## Caption options

- **Bounce, Hype, and Clean:** Choose the font and highlight color.
- **Caption position:** Place captions near the top, middle, or lower third.
- **Words per phrase:** Choose one to four words per caption group. Pauses and sentence punctuation can also end a group early.
- **Caption size:** Adjust the caption text size.
- **Uppercase captions:** Convert recognized words to uppercase.
- **Highlight spoken word:** Color the current spoken word.
- **Pop animation:** Add a subtle scale animation to the highlighted word.
- **Video frame:** Keep the source frame, or make a vertical 9:16 video with a blurred background fill.

English speech is the recommended input. Recognition quality depends on audio clarity, speaker accents, background noise, and the speech model's accuracy. Review the output before publishing.

## How it works

1. The browser uploads the selected video to the local app.
2. FFprobe checks the video's streams, duration, and dimensions. The app extracts audio for transcription.
3. Sherpa-ONNX runs the local Parakeet model and returns recognized words with timestamps. Long audio is processed in 24-second chunks; chunks below a low audio-level threshold are skipped.
4. The app groups the words according to the chosen phrase length, pause timing, and punctuation.
5. FFmpeg and its `ass` subtitle filter render the captions into the video. Captions are burned into the MP4, so they remain visible in any video player. The app also writes an SRT subtitle file.

## Files and privacy

Job data is stored under `data/jobs/<job-id>/`. A completed job folder contains the uploaded source, job information, the generated ASS and SRT subtitle files, and the captioned MP4. A locally downloaded speech model is stored under `data/models/` unless the app is reusing the neighboring PeakCut model or `CAPTION_MODEL_DIR` points elsewhere.

These folders can grow quickly because they contain source videos and rendered copies. Back up any outputs you want to keep. Deleting a job folder permanently removes that job's source and outputs; deleting a local model means it must be downloaded again before the next transcription.

The app runs its web server on loopback (`127.0.0.1`) by default. It downloads Python packages during setup and may download the speech model on first use, but it does not send uploaded videos or transcripts to a remote AI service.

## Configuration

Set environment variables before starting `./run.sh` to change the defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAPTION_PORT` | `8766` | Local web server port |
| `CAPTION_MAX_UPLOAD_GB` | `20` | Maximum upload size in GB |
| `CAPTION_MODEL_DIR` | Automatic | Path to a compatible Parakeet model folder |

Example:

```sh
CAPTION_PORT=8770 CAPTION_MAX_UPLOAD_GB=10 ./run.sh
```

Then open `http://127.0.0.1:8770`.

## Troubleshooting

### The app says FFmpeg needs subtitle support

Install the full Homebrew FFmpeg build, then stop and restart the app:

```sh
brew install ffmpeg-full
./run.sh
```

### The app cannot find or download the speech model

Check that this computer has an internet connection for the initial download and that there is enough free disk space. If PeakCut is installed, confirm its model folder is `peakcut/data/models/<model-name>`. You can also set `CAPTION_MODEL_DIR` to a compatible model folder.

### A video fails to upload or render

Check the upload limit, available disk space, and that the video contains both an audio and a video stream. If rendering fails, confirm that the installed FFmpeg includes the `ass` filter. The failed job's details appear under **Your videos**.

### The app page does not load

Make sure the Terminal window running `./run.sh` is still open, then refresh `http://127.0.0.1:8766` (or the port you configured).

## Development

The Python dependencies are listed in `requirements.txt`. To install them manually in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

`run.sh` is the recommended launcher on macOS because it also adds Homebrew's `ffmpeg-full` binary directory to `PATH` when present.

## GitHub and licenses

The `.gitignore` excludes `.venv/` and `data/`, so local virtual environments, uploaded videos, exports, and downloaded speech models are not included in a Git commit. Review `git status` before publishing to confirm no private media or model files are staged.

Caption-rendering code is adapted in part from [PeakCut](https://github.com/haziqahmedqasmi/PeakCut), under the MIT License. Font license notices are included beside the bundled fonts in `assets/fonts/`.
