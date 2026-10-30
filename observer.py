"""
observer.py - Visual Perception and Auxiliary Analysis Layer
Implements:
- Windows Per-Monitor DPI Aware v2 initialization
- Fast MSS screen region capture
- Same-dimension image quality compression (strictly preserving dimensions)
- RapidOCR offline text detection + OpenCV candidate contour reference extraction
- Frame-to-frame diff verification for anti-window-switch protection
"""

import os
import sys
import ctypes
from typing import Dict, List, Any, Optional, Tuple

import cv2
import mss
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def init_dpi_awareness() -> bool:
    """
    Enables Windows Per-Monitor DPI Aware v2 prior to UI creation.
    Calls SetProcessDpiAwarenessContext(-4) on Windows.
    Safely no-ops on macOS and Linux.
    """
    if sys.platform == "win32":
        try:
            # -4 corresponds to DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            dpi_context = ctypes.c_void_p(-4)
            result = ctypes.windll.user32.SetProcessDpiAwarenessContext(dpi_context)
            return bool(result)
        except Exception:
            try:
                # Fallback to SetProcessDpiAwareness(2) for legacy Win8.1/Win10
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
                return True
            except Exception:
                return False
    return False


class Observer:
    """
    Observer responsible for capturing the designated screen ROI,
    performing quality-only compression without resizing, and generating
    OCR + geometric auxiliary reference data.
    """

    def __init__(self, rapid_ocr_instance: Optional[RapidOCR] = None):
        """
        Initialize the Observer with an optional reusable RapidOCR instance.
        """
        self.browser_session = None
        self._ocr = rapid_ocr_instance or RapidOCR()
        self._sct = mss.mss()

    def capture_region(self, region: Dict[str, int]) -> np.ndarray:
        """
        Capture the specified region from the screen using MSS.

        :param region: Dict with keys 'left', 'top', 'width', 'height'
        :return: BGR numpy image array
        """
        if getattr(self, "browser_session", None):
            return self.browser_session.capture()
        monitor = {
            "top": int(region["top"]),
            "left": int(region["left"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        }
        raw_shot = self._sct.grab(monitor)
        # MSS returns BGRA; convert to BGR for OpenCV / model processing
        img_np = np.array(raw_shot, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        return img_bgr

    def compress_image_same_dimension(
        self,
        image: np.ndarray,
        quality: int = 85,
        output_path: Optional[str] = None
    ) -> Tuple[bytes, Tuple[int, int]]:
        """
        Compresses image encoding quality without changing image width or height.
        Strictly prohibits any downscaling or dimension resizing.

        :param image: Source BGR image numpy array
        :param quality: JPEG quality parameter (1-100), default 85
        :param output_path: Optional path to save compressed image
        :return: (compressed_bytes, (width, height))
        """
        orig_h, orig_w = image.shape[:2]
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
        success, encoded_buf = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise RuntimeError("Failed to encode image to JPEG format.")

        compressed_bytes = encoded_buf.tobytes()

        # Empirical self-check: verify decoded dimensions exactly equal original dimensions
        decoded = cv2.imdecode(np.frombuffer(compressed_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        dec_h, dec_w = decoded.shape[:2]
        if (dec_w, dec_h) != (orig_w, orig_h):
            raise ValueError(
                f"Dimension mismatch detected during compression! "
                f"Original: ({orig_w}x{orig_h}), Decoded: ({dec_w}x{dec_h})"
            )

        if output_path:
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(compressed_bytes)

        return compressed_bytes, (orig_w, orig_h)

    def detect_correct_continue(self, image: np.ndarray) -> Optional[list]:
        """Find graded feedback in the footer, caching verified label templates.

        No OCR runs unless a large green feedback banner is present. Subsequent
        matching banners use the cached Correct/Continue patches instead of OCR.
        """
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 100, 90), (95, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * 0.5 or h < 35 or y < height * 0.5:
                continue
            banner = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
            templates = getattr(self, "_feedback_templates", None)
            if templates:
                matches = []
                for patch in templates:
                    if patch.shape[0] > h or patch.shape[1] > w:
                        break
                    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED))
                    if score < 0.96:
                        break
                    matches.append(location)
                if len(matches) == 2 and matches[0][0] < matches[1][0]:
                    patch = templates[1]
                    return [x + matches[1][0] + patch.shape[1] // 2,
                            y + matches[1][1] + patch.shape[0] // 2]
            # One small OCR pass verifies both labels and locates the button.
            raw, _ = self._ocr(banner)
            labels = {}
            for points, text, confidence in raw or []:
                label = str(text).strip().lower().rstrip("!")
                if label in ("correct", "continue") and confidence >= 0.85:
                    pts = np.array(points, dtype=np.int32)
                    left, top = np.min(pts, axis=0)
                    right, bottom = np.max(pts, axis=0)
                    labels[label] = (max(0, left-2), max(0, top-2), min(w, right+3), min(h, bottom+3))
            if "correct" in labels and "continue" in labels:
                boxes = [labels["correct"], labels["continue"]]
                self._feedback_templates = [gray[t:b, l:r].copy() for l, t, r, b in boxes]
                left, top, right, bottom = labels["continue"]
                return [int(x + (left+right)//2), int(y + (top+bottom)//2)]
        return None

    def detect_enabled_submit(self, image: np.ndarray) -> Optional[list]:
        """Locate a blue Submit answer control, never a disabled gray button."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (95, 110, 100), (125, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x < width * .55 or y < height * .75 or w < 80 or h < 25:
                continue
            button = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(button, cv2.COLOR_BGR2GRAY)
            template = getattr(self, "_submit_template", None)
            if template is not None and template.shape == gray.shape:
                if cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[0, 0] > .96:
                    return [x+w//2, y+h//2]
            raw, _ = self._ocr(button)
            text = " ".join(str(item[1]).strip().lower() for item in raw or [] if item[2] >= .85)
            if text == "submit answer":
                self._submit_template = gray.copy()
                return [x+w//2, y+h//2]
        return None

    def verify_numeric_entry(self, image: np.ndarray, target: list, expected: str) -> bool:
        """Read only the edited answer slot before allowing a keyboard fast submit."""
        x, y = target
        height, width = image.shape[:2]
        crop = image[max(0, y-55):min(height, y+55), max(0, x-150):min(width, x+150)]
        if not crop.size:
            return False
        raw, _ = self._ocr(crop)
        for _, text, confidence in raw or []:
            value = str(text).strip().replace("−", "-").replace(" ", "")
            if confidence >= .85 and value == expected:
                return True
        return False

    def extract_auxiliary_reference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Performs RapidOCR inference and OpenCV contour detection on the image.
        Outputs a structured reference table containing text boxes, centers,
        confidence scores, and detected candidate input/clickable regions.

        :param image: BGR numpy image array
        :return: Dict with 'ocr_results', 'candidate_boxes', and 'formatted_table'
        """
        cached_frame = getattr(self, "_aux_frame", None)
        if cached_frame is not None and np.array_equal(cached_frame, image):
            return self._aux_reference
        img_h, img_w = image.shape[:2]
        ocr_results: List[Dict[str, Any]] = []

        # 1. Run RapidOCR
        try:
            raw_ocr, _ = self._ocr(image)
            if raw_ocr:
                for item in raw_ocr:
                    # item format: [box_points, text, confidence]
                    # box_points is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    pts = np.array(item[0], dtype=np.int32)
                    text = str(item[1]).strip()
                    conf = float(item[2])

                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))

                    ocr_results.append({
                        "text": text,
                        "box": pts.tolist(),
                        "center": [cx, cy],
                        "confidence": round(conf, 3),
                    })
        except Exception as e:
            ocr_results.append({
                "text": f"[OCR Error: {e}]",
                "box": [[0, 0], [0, 0], [0, 0], [0, 0]],
                "center": [0, 0],
                "confidence": 0.0,
            })

        # 2. Run OpenCV contour extraction for candidate input boxes / buttons
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: List[Dict[str, Any]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise and entire-screen borders
            if 20 <= w < (img_w * 0.95) and 15 <= h < (img_h * 0.95):
                aspect_ratio = float(w) / max(1, h)
                if 0.2 <= aspect_ratio <= 15.0:
                    candidate_boxes.append({
                        "bbox": [x, y, w, h],
                        "center": [x + w // 2, y + h // 2]
                    })

        # Keep top most prominent candidates to prevent prompt clutter
        candidate_boxes = sorted(
            candidate_boxes,
            key=lambda b: b["bbox"][2] * b["bbox"][3],
            reverse=True
        )[:20]

        # 3. Build human/model readable Markdown reference table
        lines = [
            f"Image Dimensions: {img_w}x{img_h}",
            "### Detected OCR Text Elements:",
            "| ID | Text | Center [X, Y] | Bounding Box | Confidence |",
            "|---|---|---|---|---|"
        ]
        if ocr_results:
            for idx, item in enumerate(ocr_results):
                pts_str = str(item["box"])
                lines.append(
                    f"| T{idx+1} | {item['text']} | {item['center']} | {pts_str} | {item['confidence']} |"
                )
        else:
            lines.append("| - | No text detected | - | - | - |")

        if candidate_boxes:
            lines.append("\n### Candidate Interactive Box Regions (OpenCV):")
            lines.append("| ID | Region [X, Y, W, H] | Center [X, Y] |")
            lines.append("|---|---|---|")
            for idx, c in enumerate(candidate_boxes[:10]):
                lines.append(f"| B{idx+1} | {c['bbox']} | {c['center']} |")

        formatted_table = "\n".join(lines)

        result = {
            "image_size": [img_w, img_h],
            "ocr_results": ocr_results,
            "candidate_boxes": candidate_boxes,
            "formatted_table": formatted_table
        }
        self._aux_frame = image.copy()
        self._aux_reference = result
        return result

    @staticmethod
    def compute_frame_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Computes the visual difference ratio between two same-sized frames.
        Used to prevent misclicks caused by window switching or unexpected popups.

        :return: Difference score in range [0.0, 1.0]
        """
        if frame_a.shape != frame_b.shape:
            return 1.0

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_a, gray_b)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        return float(non_zero) / float(max(1, total_pixels))
