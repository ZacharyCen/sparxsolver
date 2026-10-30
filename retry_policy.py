"""Per-question retry accounting and durable human-review records."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def incorrect_feedback(items: list, height: int) -> bool:
    """Require an explicit grading label in the lower half, never question prose."""
    phrases = {"incorrect", "not quite", "try again", "that's not quite right", "that’s not quite right",
               "sorry, that's not correct", "sorry, that’s not correct", "your answer is incorrect"}
    return any(item.get("confidence", 1) >= .85
               and item.get("center", [0, 0])[1] > height * .5
               and item.get("text", "").strip().lower().rstrip("!.") in phrases for item in items)


def question_tabs(frame, items: list) -> tuple:
    """Identify the navy selected question tab and its immediate right neighbour."""
    tabs = []
    height, width = frame.shape[:2]
    for item in items:
        label = item.get("text", "").strip().upper()
        match = re.fullmatch(r"(\d+[A-Z])(?:\s*[✓✔×✕X])?", label)
        x, y = map(int, item.get("center", [0, 0]))
        if not match or not (0 < y < height * .35 and 0 < x < width):
            continue
        patch = frame[max(0, y-12):y+13, max(0, x-22):x+23]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        navy = float(np.mean(cv2.inRange(hsv, (100, 65, 35), (140, 255, 155)) > 0)) > .3
        tabs.append((x, match[1], [x, y], navy))
    tabs.sort()
    for index, tab in enumerate(tabs):
        if tab[3]:
            return tab[1], tabs[index+1][2] if index+1 < len(tabs) else None
    return None, None


class QuestionRetries:
    """Count each submitted answer once and save abandoned questions for review."""
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir or Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".config"))) / "AutoAnswer" / "needs_human")
        self.reset()

    def reset(self) -> None:
        self.wrong = 0
        self.pending = False
        self.feedback_latched = False
        self.attempts = []
        self.logged = False

    def submitted(self, answer: str, effort: str) -> None:
        self.pending = True
        self.attempts.append({"answer": answer, "reasoning_effort": effort})

    def observe(self, incorrect: bool) -> bool:
        if not incorrect:
            self.feedback_latched = False
            return False
        if not self.pending or self.feedback_latched:
            return False
        self.pending = False
        self.feedback_latched = True
        self.wrong += 1
        return True

    def save(self, question: str, frame) -> Path:
        """Persist one record and screenshot per abandoned question."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        screenshot = self.log_dir / f"{stamp}.png"
        if not cv2.imwrite(str(screenshot), frame):
            raise OSError("Could not save the question review screenshot")
        path = self.log_dir / "questions.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": stamp, "question": question,
                "status": "Needs Human intervention", "wrong_answers": self.wrong,
                "attempts": self.attempts, "screenshot": str(screenshot)}, ensure_ascii=False) + "\n")
        self.logged = True
        return path
