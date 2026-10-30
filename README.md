# Sparx Solver

A Python desktop assistant for Sparx Maths. A vision model plans answer entry while local controls verify progress, handle submissions, and enforce cancellation safeguards.

## Run locally

```bash
python -m venv .venv
# Activate .venv for your platform.
python -m pip install -r requirements.txt
python main.py
```

In Configuration, select the existing Chrome homework window or open the separate browser and sign in. Choose the number of new correct answers to complete. Esc stops automation; existing-window mode uses the desktop mouse and keyboard.

## Model configuration

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`, or keep a local `OpenAI-KEY.txt` beside the application. The default remote model is `gpt-6.1-terra` with low reasoning effort. Never commit credentials.

After two confirmed incorrect submissions on a question, reasoning increases to high. After two additional failures, the assistant saves a “Needs Human intervention” record and screenshot, then attempts a verified move to the next question. Skips do not count as correct answers. Open review logs from Configuration.

## Verification

```bash
python -m unittest test_remote_workflow test_retry_policy
python test_step4.py
```

Desktop integration requires a display and input permissions. `self_test.py` additionally checks authenticated CLI access with a live image request.

## Packaging

Use `build_windows.bat` on Windows or `bash build_mac.sh` on macOS. These scripts replace generated `build/` and `dist/` output. API keys remain local sidecar files; inspect distribution contents before sharing.

See [AGENTS.md](AGENTS.md) for module organization and contribution guidelines.
