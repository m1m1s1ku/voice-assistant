# Simple voice assistant in French

Uses MLX audio and Mistral-Small-24B-Instruct-2501-4bit through MLX-LM

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --voice alexandra.json
```

_Should only work on Apple Silicon machines, tested on MacOS Sequoia (M1 Max) with python 3.11_

## Custom voice training

Install [OuteTTS](https://github.com/edwko/OuteTTS)

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install outetts --upgrade
```

Use the `voice-train.py` given in the repository (change path to wav file and name of the speaker)

Then use the resulting json file in the main.py script `--voice voice_name.json`

## TODO

- [x] Use a local LLM instead of Mistral API
- [ ] Make prompt configurable for multilanguage support (only french for now)
- [ ] Add more voices
