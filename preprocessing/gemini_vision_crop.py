"""
Gemini Vision AI Retinal Eye Detection and Cropping Module (F-03)
Integrates Gemini Multimodal Vision API for automated retinal disk localization
and field-of-view (FOV) extraction with automatic OpenCV fallback.
"""

import os
import cv2
import json
import base64
import requests
import numpy as np
from typing import Tuple, Optional, Dict, Any

from config import Config


def _opencv_fundus_crop(img_bgr: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Fallback circular crop using OpenCV contour and minEnclosingCircle.
    Used when Gemini API is unconfigured, unreachable, or offline in rural PHCs.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        cx, cy, radius = int(cx), int(cy), int(radius)
    else:
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 2

    min_expected_radius = min(h, w) * 0.25
    if radius < min_expected_radius:
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 2

    # 1:1 square crop with black padding
    r = int(radius * 1.02)
    side = 2 * r
    square = np.zeros((side, side, 3), dtype=np.uint8)

    src_x1 = max(0, cx - r)
    src_y1 = max(0, cy - r)
    src_x2 = min(w, cx + r)
    src_y2 = min(h, cy + r)

    dst_x1 = r - (cx - src_x1)
    dst_y1 = r - (cy - src_y1)

    square[dst_y1:dst_y1 + (src_y2 - src_y1),
           dst_x1:dst_x1 + (src_x2 - src_x1)] = img_bgr[src_y1:src_y2, src_x1:src_x2]

    resized = cv2.resize(square, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # Smooth circular mask
    th, tw = target_size
    tcx, tcy = tw // 2, th // 2
    tr = int(min(th, tw) * 0.485)

    circ_mask = np.zeros((th, tw), dtype=np.uint8)
    cv2.circle(circ_mask, (tcx, tcy), tr, 255, -1)
    circ_mask_smooth = cv2.GaussianBlur(circ_mask, (11, 11), 0) / 255.0
    circ_mask_3c = np.repeat(circ_mask_smooth[:, :, np.newaxis], 3, axis=2)

    result_img = (resized * circ_mask_3c).astype(np.uint8)
    return result_img, (cx, cy, radius)


def gemini_detect_and_crop(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    api_key: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int, int], str]:
    """
    Performs retinal eye detection and circular field cropping using Gemini Vision AI.
    Strictly preserves fallback behavior: if API key is not supplied, offline, or
    encountering network latency, falls back gracefully to OpenCV contour detection.

    Returns:
        cropped_bgr: Preprocessed 1:1 square retinal image (target_size)
        circle_info: (center_x, center_y, radius) in original coordinates
        detector_used: 'gemini_vision_ai' or 'opencv_fallback'
    """
    # 1. Resolve API Key
    effective_key = (
        api_key
        or os.environ.get('GEMINI_API_KEY')
        or os.environ.get('GOOGLE_API_KEY')
        or getattr(Config, 'GEMINI_API_KEY', None)
    )

    if not effective_key:
        print("[Gemini Vision AI] API key not configured. Using certified OpenCV adaptive contour fallback.")
        cropped, info = _opencv_fundus_crop(img_bgr, target_size)
        return cropped, info, 'opencv_fallback'

    try:
        # Encode image to JPEG base64 for API payload
        success, encoded_img = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            raise ValueError("Failed to encode retinal image for Gemini payload.")

        b64_data = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        h, w = img_bgr.shape[:2]

        # Construct Gemini API Request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={effective_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"Analyze this retinal fundus photograph of dimensions {w}x{h}. "
                                "Identify the circular retinal boundary (field of view). "
                                "Output JSON ONLY with keys: "
                                "{\"center_x\": int, \"center_y\": int, \"radius\": int} "
                                "in pixel coordinates relative to original image size."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        response = requests.post(url, json=payload, timeout=8.0)
        if response.status_code == 200:
            result_json = response.json()
            raw_text = result_json['candidates'][0]['content']['parts'][0]['text']
            parsed = json.loads(raw_text)

            cx = int(parsed.get('center_x', w // 2))
            cy = int(parsed.get('center_y', h // 2))
            radius = int(parsed.get('radius', min(h, w) // 2))

            # Sanity validation
            if (radius > min(h, w) * 0.20) and (0 <= cx < w) and (0 <= cy < h):
                print(f"[Gemini Vision AI] Retinal disk detected by Gemini: ({cx}, {cy}), r={radius}")
                
                # Apply 1:1 square crop with black padding
                r = int(radius * 1.02)
                side = 2 * r
                square = np.zeros((side, side, 3), dtype=np.uint8)

                src_x1 = max(0, cx - r)
                src_y1 = max(0, cy - r)
                src_x2 = min(w, cx + r)
                src_y2 = min(h, cy + r)

                dst_x1 = r - (cx - src_x1)
                dst_y1 = r - (cy - src_y1)

                square[dst_y1:dst_y1 + (src_y2 - src_y1),
                       dst_x1:dst_x1 + (src_x2 - src_x1)] = img_bgr[src_y1:src_y2, src_x1:src_x2]

                resized = cv2.resize(square, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

                # Smooth mask
                th, tw = target_size
                tcx, tcy = tw // 2, th // 2
                tr = int(min(th, tw) * 0.485)

                circ_mask = np.zeros((th, tw), dtype=np.uint8)
                cv2.circle(circ_mask, (tcx, tcy), tr, 255, -1)
                circ_mask_smooth = cv2.GaussianBlur(circ_mask, (11, 11), 0) / 255.0
                circ_mask_3c = np.repeat(circ_mask_smooth[:, :, np.newaxis], 3, axis=2)

                result_img = (resized * circ_mask_3c).astype(np.uint8)
                return result_img, (cx, cy, radius), 'gemini_vision_ai'

        print(f"[Gemini Vision AI] API call returned non-200 ({response.status_code}). Triggering fallback.")
    except Exception as exc:
        print(f"[Gemini Vision AI] Exception during API call ({exc}). Smoothly falling back to OpenCV detector.")

    # Seamless Fallback
    cropped, info = _opencv_fundus_crop(img_bgr, target_size)
    return cropped, info, 'opencv_fallback'
