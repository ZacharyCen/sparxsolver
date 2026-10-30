"""
planner.py - Dual-Mode Decision Layer
Implements:
1. CodexCliPlanner (Key-Free Mode: codex_dev)
   - Zero-key codex exec subprocess invocation with argument array
   - CREATE_NO_WINDOW isolation under sys.platform == "win32"
   - Three mandatory startup checks:
     * Check 1: Executable discovery
     * Check 2: Authentication status verification
     * Check 3: Real image request dry-run round-trip check
2. RemoteApiPlanner (External API Distribution Mode: remote)
   - OpenAI-compatible multimodal vision API invoker skeleton
   - Strict local schema validation against action.schema.json
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import jsonschema


def get_default_schema_path() -> str:
    """Returns the default absolute path to action.schema.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "action.schema.json")
    if not os.path.isfile(schema_path):
        # Fallback for PyInstaller frozen bundle
        if hasattr(sys, "_MEIPASS"):
            schema_path = os.path.join(sys._MEIPASS, "action.schema.json")
    return schema_path


class BasePlanner(ABC):
    """Abstract base class for action planners."""

    @abstractmethod
    def plan(
        self,
        frame_path: str,
        question_state: str,
        auxiliary_reference: str,
        previous_result: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Given the captured frame and auxiliary context, plan the next single-step action.
        """
        pass


class CodexCliPlanner(BasePlanner):
    """
    Key-free local Codex CLI subprocess invoker (codex_dev mode).
    Reuses authenticated local CLI session without configuring external API keys.
    """

    def __init__(self, codex_bin: Optional[str] = None, schema_path: Optional[str] = None):
        self.schema_path = schema_path or get_default_schema_path()
        self.reasoning_effort = "low"
        self.question_memory = ""
        self.recent_steps = []
        self.retry_context = ""
        self._schema_data: Optional[Dict[str, Any]] = None
        self._load_schema()

        # Locate executable
        self.codex_bin = codex_bin or self.find_codex_binary()

    def reset_question_memory(self) -> None:
        """Reset retry instructions and reasoning for the next question."""
        self.reasoning_effort = "low"
        self.question_memory = self.retry_context = ""
        self.recent_steps = []

    def _load_schema(self) -> Dict[str, Any]:
        if not os.path.isfile(self.schema_path):
            raise FileNotFoundError(f"Action schema file not found at: {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self._schema_data = json.load(f)
        return self._schema_data

    @staticmethod
    def find_codex_binary() -> str:
        """
        Locates the codex executable in the environment.
        Checks PATH and common platform installation locations.
        """
        # 1. Check environment variable override
        env_bin = os.environ.get("CODEX_BIN") or os.environ.get("CODEX_PATH")
        if env_bin and os.path.isfile(env_bin):
            return env_bin

        # 2. Check standard PATH
        binary_names = ["codex.cmd", "codex.exe", "codex"] if sys.platform == "win32" else ["codex"]
        for name in binary_names:
            resolved = shutil.which(name)
            if resolved:
                return resolved

        # 3. Check common local paths
        home = os.path.expanduser("~")
        candidate_paths = [
            os.path.join(home, ".local", "bin", "codex"),
            os.path.join(home, "AppData", "Roaming", "npm", "codex.cmd"),
            os.path.join(home, "AppData", "Local", "Programs", "codex", "codex.exe"),
            "/usr/local/bin/codex",
            "/opt/homebrew/bin/codex",
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                return p

        raise FileNotFoundError(
            "Could not locate 'codex' executable. Please ensure OpenAI Codex CLI is "
            "installed ('npm install -g @openai/codex') and accessible in PATH."
        )

    # =========================================================================
    # Three Mandatory Startup Checks (per Section III.1)
    # =========================================================================

    def check_executable(self) -> str:
        """
        Check 1 (Executable Check):
        Verifies that a valid codex executable path is located and callable.
        """
        if not self.codex_bin or not os.path.isfile(self.codex_bin):
            self.codex_bin = self.find_codex_binary()

        # Test simple version flag
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            res = subprocess.run(
                [self.codex_bin, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
                timeout=10
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"Codex executable verification failed with exit code {res.returncode}: {res.stderr}"
                )
            return self.codex_bin
        except Exception as e:
            raise RuntimeError(f"Failed to execute codex binary '{self.codex_bin}': {e}") from e

    def check_authentication(self) -> bool:
        """
        Check 2 (Authentication Check):
        Verifies that the current runtime environment has valid CLI authentication.
        Raises an explicit error upon non-zero exit or authentication failure.
        """
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        # Run `codex doctor` to verify environment and authentication status
        try:
            res = subprocess.run(
                [self.codex_bin, "doctor"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
                timeout=15
            )
            combined_output = (res.stdout + "\n" + res.stderr).lower()
            if "no codex credentials were found" in combined_output or "unauthorized" in combined_output:
                raise RuntimeError(
                    "Codex CLI authentication check failed: No active credentials found. "
                    "Please run 'codex login' in your terminal."
                )
            return True
        except subprocess.TimeoutExpired:
            raise RuntimeError("Authentication check timed out while querying Codex status.")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Unexpected error during authentication check: {e}") from e

    def check_image_roundtrip(self, test_image_path: str) -> Dict[str, Any]:
        """
        Check 3 (Real Image Request Check):
        Validates the entire pipeline 'Account Auth -> Image Input -> Action JSON Return'
        using an actual image screenshot, with strict validation against action.schema.json.
        """
        if not os.path.isfile(test_image_path):
            raise FileNotFoundError(f"Test image for dry-run check not found: {test_image_path}")

        prompt = (
            "You are an automated quiz answering assistant. "
            "Analyze this question view image and return the single next action JSON strictly adhering to schema. "
            "State: testing round-trip validation."
        )

        action_dict = self.plan(
            frame_path=test_image_path,
            question_state="testing",
            auxiliary_reference="Dry-run startup check image",
            previous_result=None
        )

        # Validate against schema
        self.validate_action(action_dict)
        return action_dict

    def run_startup_self_check(self, test_image_path: str) -> bool:
        """
        Executes all three mandatory startup checks in sequence.
        """
        print("[Planner] Running Startup Check 1/3: Executable Check...")
        bin_path = self.check_executable()
        print(f"[Planner] Check 1 Passed: Located executable at '{bin_path}'")

        print("[Planner] Running Startup Check 2/3: Authentication Check...")
        self.check_authentication()
        print("[Planner] Check 2 Passed: CLI authentication session verified.")

        print("[Planner] Running Startup Check 3/3: Real Image Round-Trip Check...")
        action_payload = self.check_image_roundtrip(test_image_path)
        print(f"[Planner] Check 3 Passed: Successfully received valid action payload: {action_payload.get('action')}")
        return True

    # =========================================================================
    # Plan Execution via Subprocess
    # =========================================================================

    def plan(
        self,
        frame_path: str,
        question_state: str,
        auxiliary_reference: str,
        previous_result: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calls `codex exec` non-interactively using parameter arrays and file redirection.
        """
        if not os.path.isfile(frame_path):
            raise FileNotFoundError(f"Input image frame not found: {frame_path}")

        from PIL import Image
        with Image.open(frame_path) as image:
            frame_width, frame_height = image.size

        prompt_text = (
            f"Original image size: {frame_width} x {frame_height} pixels. All action coordinates must use this coordinate system.\n"
            f"You are an autonomous homework answering assistant on Sparx Maths.\n"
            f"Current Question State: {question_state} (new correct answers in THIS run; ignore previously completed sidebar items).\n"
            f"Complete every part of a multipart question before submission. A filled part a is not a completed question. Never infer the run target was met from existing checkmarks; the controller owns that counter.\n"
            f"Retry instructions: {self.retry_context}\n"
            f"Previous Step Result: {previous_result or 'None (initial step)'}\n\n"
            f"Auxiliary Detection Reference Table:\n{auxiliary_reference}\n\n"
            f"RULES & WORKFLOW:\n"
            f"1. INITIAL QUESTION VIEW: If a question is displayed and the answer input / virtual keypad is not yet revealed, look for the 'Answer' button (typically orange button at bottom-right) or the answer area, and click it (type='click', purpose='answer'). The page_state is 'question', NOT 'blocked'.\n"
            f"2. SOLVING & INPUT: When the answer box, keypad, or multiple-choice option cards are visible, solve the math problem with mathematical precision.\n"
            f"   - For numerical/text input: use type='fill' with the exact answer into the input field, or click the keypad keys.\n"
            f"   - For multiple choice cards: click the card matching the mathematically correct answer (type='click', purpose='answer').\n"
            f"3. SUBMISSION: Once the answer has been entered and 'Submit answer' is visible and active, click 'Submit answer' (type='click', purpose='submit').\n"
            f"4. ADVANCING: If the screen displays 'Correct!' or feedback with a 'Continue' button, click 'Continue' (type='click', purpose='next') to advance to the next question.\n"
            f"5. SAFETY: Only return page_state='blocked' or type='stop' if a true blocker (e.g. CAPTCHA, crash dialog, login required) prevents answering.\n\n"
            f"COORDINATES: Use pixels in the original attached image, origin top-left; use OCR centers to locate controls. Never infer coordinates from a resized preview.\n"
            f"VERIFICATION: Previous results are observations, not proof of success. If a click had no confirmed effect, re-locate the target. Do not repeat an ineffective click blindly.\n"
            f"Only use purpose='next' on actual result feedback with a Continue/next-question control, and set page_state='feedback'. A button that returns to question text is NOT question completion.\n"
            f"Instruction: Return ONLY the single next atomic action JSON strictly adhering to schema."
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out:
            output_json_path = tmp_out.name

        try:
            # Strictly use parameter arrays; pass prompt via stdin ("-") to prevent Windows command-line length limits (8191 chars)
            cmd = [
                self.codex_bin, "exec",
                "--ephemeral",
                "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--image", os.path.abspath(frame_path),
                "--output-schema", os.path.abspath(self.schema_path),
                "-o", os.path.abspath(output_json_path),
                "-"
            ]

            # Platform Process Flag Isolation: CREATE_NO_WINDOW only on win32
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

            res = subprocess.run(
                cmd,
                input=prompt_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                timeout=60
            )

            if res.returncode != 0:
                raise RuntimeError(
                    f"Codex subprocess exited with error code {res.returncode}.\n"
                    f"Stderr: {res.stderr}\nStdout: {res.stdout}"
                )

            if not os.path.isfile(output_json_path) or os.path.getsize(output_json_path) == 0:
                raise RuntimeError("Codex did not produce the expected action output JSON file.")

            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "frame_id" not in data or not data["frame_id"]:
                data["frame_id"] = os.path.splitext(os.path.basename(frame_path))[0]
            self.validate_action(data)
            return data

        finally:
            if os.path.isfile(output_json_path):
                try:
                    os.remove(output_json_path)
                except OSError:
                    pass

    def validate_action(self, action_dict: Dict[str, Any]) -> None:
        """
        Strictly validates action dictionary against action.schema.json using jsonschema.
        """
        if self._schema_data is None:
            self._load_schema()
        jsonschema.validate(instance=action_dict, schema=self._schema_data)


def load_api_config(config_path: Optional[str] = None) -> Dict[str, str]:
    """Load replacement OpenAI credentials without using the legacy key file."""
    import re
    config = {"api_key": os.environ.get("OPENAI_API_KEY", ""),
              "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
              "model": os.environ.get("OPENAI_MODEL", "gpt-6.1-terra")}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    roots = ([os.path.dirname(sys.executable)] if getattr(sys, "frozen", False) else []) + [base_dir, os.getcwd()]
    if hasattr(sys, "_MEIPASS"):
        roots.append(sys._MEIPASS)
    paths = ([config_path] if config_path else []) + [os.path.join(root, "OpenAI-KEY.txt") for root in roots]
    for path in dict.fromkeys(paths):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as stream:
            content = stream.read().strip()
        if content.startswith("{"):
            values = json.loads(content)
            for key in config:
                if values.get(key):
                    config[key] = str(values[key]).strip()
        else:
            for line in content.splitlines():
                if re.fullmatch(r"sk-[\w-]+", line.strip()):
                    config["api_key"] = line.strip()
                for key, pattern in (("api_key", r"(?:OPENAI_)?API[ _-]*KEY\s*[:=：]\s*(\S+)"),
                                     ("base_url", r"(?:OPENAI_)?BASE[ _-]*URL\s*[:=：]\s*(\S+)"),
                                     ("model", r"(?:OPENAI_)?MODEL\s*[:=：]\s*(\S+)")):
                    match = re.search(pattern, line, re.I)
                    if match:
                        config[key] = match.group(1).strip("\"'")
        break
    return config


class RemoteApiPlanner(BasePlanner):
    """
    Remote external vision model API invoker (remote distribution mode).
    Compatible with OpenAI multimodal endpoints (e.g. gpt-4o / gpt-6.1-terra).
    Crops the question card at native resolution, maps coordinates back to the
    captured frame, and validates every response against the local action schema.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        schema_path: Optional[str] = None
    ):
        defaults = load_api_config()
        self.api_key = api_key or defaults["api_key"]
        self.base_url = base_url or defaults["base_url"]
        self.model = model or defaults["model"]
        self.recent_steps = []
        self.question_memory = ""
        self.last_usage = {}
        self.usage_callback = None
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        self.keyboard_entry_enabled = True
        self.reasoning_effort = "low"
        self.retry_context = ""
        self.schema_path = schema_path or get_default_schema_path()

        if not self.api_key:
            raise ValueError("API Key is required for RemoteApiPlanner. Please provide OpenAI-KEY.txt or OPENAI_API_KEY.")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema_data = json.load(f)

        def compact_schema(value):
            if isinstance(value, dict):
                return {k: compact_schema(v) for k, v in value.items()
                        if k not in ("description", "title", "$schema")}
            if isinstance(value, list):
                return [compact_schema(v) for v in value]
            return value
        self.request_schema = compact_schema(self.schema_data)
        # The API requires every root property; local validation accepts older decisions.
        self.request_schema["required"] = list(self.request_schema["properties"])

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def reset_question_memory(self) -> None:
        """Discard the previous solution after verified navigation to a new question."""
        self.question_memory = ""
        self.recent_steps = []
        self.reasoning_effort = "low"
        self.retry_context = ""

    def plan(
        self,
        frame_path: str,
        question_state: str,
        auxiliary_reference: str,
        previous_result: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calls remote external vision model API and validates returned JSON locally.
        """
        import base64
        import time

        if not os.path.isfile(frame_path):
            raise FileNotFoundError(f"Input image frame not found: {frame_path}")

        import io
        import cv2
        import numpy as np
        from PIL import Image
        with Image.open(frame_path) as source:
            source = source.convert("RGB")
            original_width, original_height = source.size
            pixels = np.asarray(source)
            mask = cv2.inRange(pixels, (248, 248, 248), (255, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours]
            cards = [(x, y, w, h) for x, y, w, h in boxes
                     if w > original_width * .5 and h > original_height * .45
                     and w < original_width * .95 and y > original_height * .05]
            left = top = 0
            right, bottom = original_width, original_height
            if cards:
                x, y, w, h = max(cards, key=lambda box: box[2] * box[3])
                left, top = max(0, x-12), max(0, y-12)
                right = min(original_width, x+w+12)
                # Keep all lower feedback/submit controls, including colored footers.
            cropped = source.crop((left, top, right, bottom))
            width, height = cropped.size
            buffer = io.BytesIO()
            cropped.save(buffer, format="JPEG", quality=85)
            b64_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        if left or top:
            print(f"[Image] Question card: {width}x{height} (from {original_width}x{original_height}); coordinate offset ({left}, {top})")
        try:
            items = json.loads(auxiliary_reference)
            if isinstance(items, list):
                grounded = []
                for item in items:
                    x, y = item["center"]
                    if left <= x < right and top <= y < bottom:
                        grounded.append([item["text"], x-left, y-top])
                auxiliary_reference = json.dumps(grounded, ensure_ascii=False, separators=(",", ":"))
        except (ValueError, TypeError, KeyError):
            # Legacy formatted tables use full-frame coordinates; avoid mixing systems.
            if left or top:
                auxiliary_reference = ""
        history = "\n".join(self.recent_steps[-2:]) or "None"
        prompt_text = (
            f"Sparx Maths desktop controller. Image: {width}x{height}; coordinates are original-image pixels from top-left.\n"
            f"Session: {question_state}. Count only new correct answers; controller owns the limit, not sidebar checkmarks.\n"
            f"Saved question/solution: {self.question_memory or 'None'}\n"
            f"Retry instructions: {self.retry_context}\n"
            f"Recent actions: {history}\nResult: {previous_result or 'Initial'}\n"
            f"Current OCR centers (optional): {auxiliary_reference}\n"
            "Solve ALL parts before opening Answer; record the full solution concisely in answer_display. "
            + ("After incorrect feedback, return to Question if needed and re-solve; do not trust a rejected solution. " if self.retry_context else
             "The answer page hides the question: use saved solution; never navigate Back/Previous/Question to recover it. ")
            + ("For numeric fields/custom keypads, prefer type_text to enter the whole number via physical keys in one action. "
             if self.keyboard_entry_enabled else "Keyboard entry is disabled; use visible keypad keys one per action. ")
            + "Use fill for normal text inputs. If keyboard entry is rejected, fall back to visible keypad clicks. "
            "Verify the actual entered value; do not repeat ineffective fills/clicks. "
            "Submit only when all parts are filled. Use next only on graded feedback with Continue, page_state=feedback. "
            "Page transitions: wait only if still loading. Stop on real blockers (login/CAPTCHA/crash), not ordinary question controls. "
            "Set ready_to_submit=true only when this action fills the LAST remaining answer completely. False for focusing, opening Answer, or intermediate keypad digits. "
            "Return one schema-valid atomic action; concise answer_display and expected_change."
        )

        t0 = time.time()
        print(f"[RemoteApiPlanner] Calling model '{self.model}' at {self.base_url}...")
        response = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "action_decision",
                    "strict": True,
                    "schema": self.request_schema
                }
            }
        )
        latency = time.time() - t0
        usage = getattr(response, "usage", None)
        self.last_usage = usage.model_dump() if usage and hasattr(usage, "model_dump") else {}
        reported = isinstance(self.last_usage.get("prompt_tokens"), int) and isinstance(self.last_usage.get("completion_tokens"), int)
        record = {"input_tokens": self.last_usage.get("prompt_tokens", 0) or 0,
                  "output_tokens": self.last_usage.get("completion_tokens", 0) or 0,
                  "available": reported, "seconds": latency}
        self.total_usage["requests"] += 1
        if reported:
            self.total_usage["input_tokens"] += record["input_tokens"]
            self.total_usage["output_tokens"] += record["output_tokens"]
            print(f"[Usage] Input: {record['input_tokens']}; output: {record['output_tokens']} tokens")
        if self.usage_callback:
            try:
                self.usage_callback(record)
            except Exception:
                print("[Usage] The usage display could not be updated.")

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Remote API returned empty response content.")

        parsed = json.loads(content)
        jsonschema.validate(instance=parsed, schema=self.schema_data)
        for key in ("target", "start", "end"):
            if key in parsed["action"]:
                x, y = parsed["action"][key]
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f"Model {key} outside supplied image")
                parsed["action"][key] = [x + left, y + top]
        if not self.question_memory:
            self.question_memory = parsed.get("answer_display", "")
        self.recent_steps.append(json.dumps({k: v for k, v in parsed["action"].items() if k not in ("target", "start", "end")}, ensure_ascii=False, separators=(",", ":")))
        self.recent_steps = self.recent_steps[-2:]
        act_info = parsed.get("action", {})
        print(f"[RemoteApiPlanner] Response in {latency:.2f}s: type='{act_info.get('type')}', purpose='{act_info.get('purpose')}'")
        return parsed
