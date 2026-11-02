# Sparx Solver

A Python 3.10+ desktop assistant for Sparx Maths. A vision model plans answer entry while local controls verify progress, handle submissions, and enforce cancellation safeguards.

## Run locally

```bash
python -m venv .venv
# Activate .venv for your platform.
python -m pip install -r requirements.txt
python main.py
```

On Solver, select the existing Chrome homework window or open the separate browser and sign in. Configuration contains the number of new correct answers to complete and advanced pacing settings. Esc stops automation; existing-window mode uses the desktop mouse and keyboard.

## Bookwork checks (initial numeric support)

Starting without a homework record now prompts you to create one. On Solver, choose **New homework** before starting a different assignment, or **Open record** to resume that same assignment. Use a separate record for every homework, even when question codes repeat.

On Solver, leave **This homework requires a bookwork check** checked for homework with an end check. After all questions in the detected set, the run stays active, opens Summary, selects the unique matching saved answer, submits, verifies **Passed**, and clicks Continue before reporting completion. Uncheck it for exercises without a check.

The assistant remembers numeric answers only after input readback, a dispatched submission, and confirmed correct feedback. It matches both single-number cards and complete numeric answer-image cards, including simultaneous-equation root pairs displayed in either order. Bookwork matching uses local OCR and exact numeric comparison, without a model request. Bookwork checks do not count as solved maths questions or consume ordinary retry attempts.

Missing answers, unreadable or ambiguous choices, unsupported nonnumeric/visual answers, and unconfirmed results stop for human review. Previously completed answers cannot be reconstructed from checkmarks. **Review pending** discards uncertain saved outcomes after you inspect the browser; it never marks them correct. Records stay in the local application data directory.

## Automatic question sets

The Solver detects the full tab row (for example, 1A–1G) and works through unfinished questions without a manual count. Previously correct tabs are skipped. The run ends at the set summary, with bookwork completed when required. Unreadable or changing question sets stop with a visible explanation.

## Minimal workings

On Solver, use **Generate workings for this question set** to enable or disable workings before starting. Turning it off skips model-generated derivations and automatic Markdown checkpoints, while retaining numeric answers for bookwork checks. Existing files remain available, and the choice is saved on this device.

Use **Workings → Save Markdown** to receive a local `.md` file, **Copy LaTeX** to copy its contents, or **Open folder** to find the automatic checkpoint. Verified geometry choices export as diagrams or choice letters with a portable image-assets folder beside the Markdown file. New supported numeric solutions include minimal, word-free LaTeX in the existing solving request; saving and viewing make no extra model requests.

Only the latest verified submitted snapshot with supported workings is exported. Navigation responses cannot erase an existing calculation by returning null workings; a run started on an empty answer keypad may return to read the hidden question once. The page shows captured-question coverage and reasons for review, including older answers without saved workings and unsupported visual tasks. Numeric multipart questions are saved only after every visible answer field matches the complete solution and the submission is graded correct. The numeric checks verify supported syntax, arithmetic equalities, and the final submitted value; they do not prove arbitrary symbolic derivations.

Checkpoints update outside the repository and remain available after stopping. Save As writes a consistent copy in the background. This release displays LaTeX source; rendered mathematical preview and PDF are later steps.

## Model configuration

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`, or keep a local `OpenAI-KEY.txt` beside the application. The default remote model is `gpt-6.1-terra` with low reasoning effort. Never commit credentials.

Saved single- and multi-select letter answers are matched to the current buttons locally; extra selections are removed and all required choices are confirmed before submission. Auto mode reuses tab/footer grounding instead of repeating full-screen OCR, and page transitions settle locally instead of asking the model to wait.

Numeric entry uses local field readback and a recognized keypad layout to avoid a model call per digit. Decimal points and all answer fields are checked before submission. Repeated waits, Back/Answer oscillation, and failed readback stop with an explicit review reason.

After the first confirmed incorrect submission on a question, reasoning increases to high. After four total failures, the assistant saves a “Needs Human intervention” record and screenshot, then attempts a verified move to the next question. Skips do not count as correct answers. Open review logs from Configuration.

## Verification

```bash
python -m unittest test_remote_workflow test_retry_policy test_bookwork test_workings_export test_solver_recovery test_bookwork_finish test_question_set test_choice_optimizations
python test_step4.py
```

Desktop integration requires a display and input permissions. `self_test.py` additionally checks authenticated CLI access with a live image request.

## Packaging

Use `build_windows.bat` on Windows or `bash build_mac.sh` on macOS. These scripts replace generated `build/` and `dist/` output. API keys remain local sidecar files; inspect distribution contents before sharing.

See [AGENTS.md](AGENTS.md) for module organization and contribution guidelines.
