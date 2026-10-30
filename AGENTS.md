# Repository Guidelines

## Project Structure & Module Organization

This Python desktop automation application keeps its modules at the repository root:
- `main.py`: CustomTkinter interface, region selection, and emergency stop listener.
- `controller.py`: observe–plan–validate–execute–verify control loop.
- `observer.py`: screen capture, OCR, and image comparison.
- `planner.py`: local CLI and remote API planners; `action.schema.json` defines their action contract.
- `executor.py`: mouse, keyboard, and clipboard actions; `win_utils.py` handles window integration.

Acceptance checks live in `self_test.py` and `test_step4.py`. Root-level PNG/JPG files are debugging captures. `build/` and `dist/` contain generated packaging output; do not edit bundled dependencies.

## Build, Test, and Development Commands

Run commands from the repository root using your environment’s Python executable:
- `python -m venv .venv`: create an environment; activate it before installing dependencies.
- `python -m pip install -r requirements.txt`: install dependencies, preserving NumPy’s `<2.0.0` constraint.
- `python main.py`: launch the desktop interface.
- `python test_step4.py`: run five safety acceptance checks.
- `python self_test.py`: check dependencies, schema validation, and an authenticated CLI image round trip.
- `bash build_mac.sh` on macOS or `build_windows.bat` on Windows: package with PyInstaller. Both clear previous `build/` and `dist/` output.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions and variables, and `PascalCase` classes. Add type annotations and short docstrings for public interfaces. Keep observation, planning, and physical execution in their respective modules. No formatter or linter configuration is provided.

## Testing Guidelines

Tests use plain assertions and standalone runners; no coverage threshold is configured. Name new checks `test_<behavior>`. Extend schema checks when changing actions. Verify cancellation, coordinate offsets, submission deduplication, and window-change protection when modifying execution. Desktop checks need a display and input permissions; `self_test.py` also needs authenticated CLI access and a live model request.

## Commit & Pull Request Guidelines

This checkout has no Git metadata, so commit conventions cannot be verified. Use imperative subjects such as `Fix cancellation during clipboard paste`. Include the problem, affected modules, validation results, and platform in pull requests. Link relevant issues and attach screenshots for interface changes.

## Security & Configuration

Keep credentials out of commits and screenshots. Remote configuration supports `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. Packaging scripts bundle `API KEY.txt`; inspect credential handling before distributing builds. Preserve emergency-stop and input-release safeguards.
