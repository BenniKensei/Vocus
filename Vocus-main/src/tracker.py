"""MediaPipe-based zoom and gesture tracking for Vocus.

The tracker module converts hand landmark geometry into high-level commands.
Zoom uses a calibrated openness score, while gesture mode uses a linear SVM
trained on normalized landmark-distance vectors.
"""

from __future__ import annotations

import collections
import os
import pickle
import time
import urllib.request
from types import SimpleNamespace
from typing import Optional, Tuple

import mediapipe as mp
import numpy as np

from src.config import ConfigLoader


_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class _HandDetector:
    """Compatibility wrapper for legacy and modern MediaPipe hand APIs."""

    def __init__(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._legacy_hands = None
        self._task_landmarker = None

        if hasattr(mp, "solutions"):
            mp_hands = mp.solutions.hands
            self._legacy_hands = mp_hands.Hands(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = self._ensure_task_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._task_landmarker = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_task_model() -> str:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            urllib.request.urlretrieve(_HAND_LANDMARKER_MODEL_URL, model_path)
        return model_path

    def process(self, image: np.ndarray):
        if self._legacy_hands is not None:
            return self._legacy_hands.process(image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._task_landmarker.detect(mp_image)
        wrapped_hands = [SimpleNamespace(landmark=landmarks) for landmarks in result.hand_landmarks]
        return SimpleNamespace(multi_hand_landmarks=wrapped_hands)


class ZoomTracker:
    """Track one-hand openness and map it to smooth zoom actions.

    The class uses MediaPipe hand landmarks and a calibrated openness heuristic.
    The thresholds intentionally bias against accidental zoom-out so neutral or
    partially open hands do not explode into large page zoom changes.
    """

    _WRIST = 0
    _THUMB_TIP = 4
    _INDEX_TIP = 8
    _MIDDLE_TIP = 12
    _RING_TIP = 16
    _PINKY_TIP = 20
    _INDEX_MCP = 5
    _MIDDLE_MCP = 9
    _PINKY_MCP = 17

    def __init__(self) -> None:
        """Initialize the zoom tracker and load runtime thresholds.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        cfg = ConfigLoader.get().get("zoom", {})
        self.smooth_alpha = float(cfg.get("smooth_alpha", 0.42))
        self.fist_threshold = float(cfg.get("fist_threshold", 0.56))
        self.neutral_threshold = float(cfg.get("neutral_threshold", 0.98))
        self.over_open_threshold = float(cfg.get("over_open_threshold", 1.38))
        self.deadzone_in = float(cfg.get("deadzone_in", cfg.get("deadzone_norm", 0.16)))
        self.deadzone_out = float(cfg.get("deadzone_out", 0.24))
        self.strength_curve_in = float(cfg.get("strength_curve_in", cfg.get("strength_curve", 2.1)))
        self.strength_curve_out = float(cfg.get("strength_curve_out", 2.4))
        self.out_gain = float(cfg.get("out_gain", 0.6))
        self.min_magnitude = int(cfg.get("min_magnitude", 1))
        self.max_magnitude = int(cfg.get("max_magnitude", 2))
        self.reset_cooldown_sec = float(cfg.get("reset_cooldown_sec", 1.0))
        self.hold_last_motion_sec = float(cfg.get("hold_last_motion_sec", 0.26))

        self.hands = _HandDetector(
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        )

        self._smoothed_metric: Optional[float] = None
        self._last_reset_ts = 0.0
        self._reset_pose_latched = False
        self._last_motion_cmd: Optional[Tuple[str, int]] = None
        self._last_motion_ts = 0.0
        self.last_bounding_box: Optional[Tuple[int, int, int, int]] = None
        self.last_metric: Optional[float] = None

    @staticmethod
    def _dist(a, b) -> float:
        """Compute the Euclidean distance between two MediaPipe landmarks.

        Args:
            a: First landmark.
            b: Second landmark.

        Returns:
            Scalar distance in normalized landmark space.

        Raises:
            None.
        """
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    def process_frame(self, image: np.ndarray) -> Optional[Tuple[str, int]]:
        """Convert an RGB frame into a zoom command.

        Args:
            image: RGB frame with shape ``(height, width, 3)``.

        Returns:
            ``("ZOOM_IN", magnitude)``, ``("ZOOM_OUT", magnitude)``,
            ``("RESET_ZOOM", 1)``, or ``None`` when no stable action exists.

        Raises:
            None.
        """
        h, w, _ = image.shape
        self.last_bounding_box = None
        self.last_metric = None

        results = self.hands.process(image)
        if not results.multi_hand_landmarks:
            self._smoothed_metric = None
            self._last_motion_cmd = None
            return None

        hand = results.multi_hand_landmarks[0]
        x_min = min([lm.x for lm in hand.landmark])
        x_max = max([lm.x for lm in hand.landmark])
        y_min = min([lm.y for lm in hand.landmark])
        y_max = max([lm.y for lm in hand.landmark])
        self.last_bounding_box = (
            int(x_min * w), int(y_min * h),
            int(x_max * w), int(y_max * h),
        )

        lm = hand.landmark

        # Open-hand palm pose (all fingers extended, thumb tucked) triggers browser zoom reset (Ctrl+0).
        # Latch prevents repeated resets while holding the same pose.
        reset_pose = self._detect_reset_pose(lm)
        if reset_pose:
            if not self._reset_pose_latched:
                now = time.monotonic()
                if now - self._last_reset_ts >= self.reset_cooldown_sec:
                    self._last_reset_ts = now
                    self._reset_pose_latched = True
                    self._last_motion_cmd = None
                    return "RESET_ZOOM", 1
            return None
        self._reset_pose_latched = False

        thumb = lm[self._THUMB_TIP]
        tip_ids = [self._INDEX_TIP, self._MIDDLE_TIP, self._RING_TIP, self._PINKY_TIP]
        thumb_to_fingers = [self._dist(thumb, lm[i]) for i in tip_ids]

        # Normalize by hand scale so thresholds are robust to distance from camera.
        scale = 0.5 * (
            self._dist(lm[self._WRIST], lm[self._MIDDLE_MCP])
            + self._dist(lm[self._INDEX_MCP], lm[self._PINKY_MCP])
        )
        scale = max(scale, 1e-6)
        openness = float(np.mean(thumb_to_fingers) / scale)

        a = min(1.0, max(0.1, self.smooth_alpha))
        if self._smoothed_metric is None:
            self._smoothed_metric = openness
        else:
            self._smoothed_metric = a * openness + (1.0 - a) * self._smoothed_metric
        self.last_metric = self._smoothed_metric

        metric = self._smoothed_metric
        if metric >= self.neutral_threshold:
            denom = max(1e-6, self.over_open_threshold - self.neutral_threshold)
            signed = (metric - self.neutral_threshold) / denom
        else:
            denom = max(1e-6, self.neutral_threshold - self.fist_threshold)
            signed = -((self.neutral_threshold - metric) / denom)

        # Clamp to [-1, 1] where -1 is full fist (max zoom-in) and
        # +1 requires over-extension to reach max zoom-out.
        signed = max(-1.0, min(1.0, signed))

        if signed > 0:
            deadzone = self.deadzone_out
            curve = self.strength_curve_out
        else:
            deadzone = self.deadzone_in
            curve = self.strength_curve_in

        if abs(signed) < deadzone:
            return None

        direction = "ZOOM_OUT" if signed > 0 else "ZOOM_IN"
        strength = (abs(signed) - deadzone) / max(1e-6, 1.0 - deadzone)
        strength = max(0.0, min(1.0, strength))
        strength = strength ** curve
        if signed > 0:
            strength *= self.out_gain
        magnitude = int(round(self.min_magnitude + strength * (self.max_magnitude - self.min_magnitude)))
        magnitude = max(self.min_magnitude, min(self.max_magnitude, magnitude))
        cmd = (direction, magnitude)
        self._last_motion_cmd = cmd
        self._last_motion_ts = time.monotonic()
        return cmd

    @staticmethod
    def _is_extended(tip, pip) -> bool:
        """Check whether a fingertip is extended above its PIP joint.

        Args:
            tip: Fingertip landmark.
            pip: Proximal interphalangeal joint landmark.

        Returns:
            ``True`` when the finger is extended, otherwise ``False``.

        Raises:
            None.
        """
        return tip.y < pip.y

    def _detect_reset_pose(self, lm) -> bool:
        """Detect the open-palm reset gesture with the thumb tucked.

        Args:
            lm: MediaPipe hand landmark list.

        Returns:
            ``True`` when the reset gesture is present.

        Raises:
            None.
        """
        index_ext = self._is_extended(lm[self._INDEX_TIP], lm[6])
        middle_ext = self._is_extended(lm[self._MIDDLE_TIP], lm[10])
        ring_ext = self._is_extended(lm[self._RING_TIP], lm[14])
        pinky_ext = self._is_extended(lm[self._PINKY_TIP], lm[18])

        palm_scale = 0.5 * (
            self._dist(lm[self._WRIST], lm[self._MIDDLE_MCP])
            + self._dist(lm[self._INDEX_MCP], lm[self._PINKY_MCP])
        )
        palm_scale = max(palm_scale, 1e-6)

        thumb_tip_to_index_mcp = self._dist(lm[self._THUMB_TIP], lm[self._INDEX_MCP]) / palm_scale
        thumb_tucked = thumb_tip_to_index_mcp < 0.85

        return index_ext and middle_ext and ring_ext and pinky_ext and thumb_tucked


class GestureTracker:
    """Classify hand gestures into scroll actions with a linear SVM.

    The model consumes normalized landmark-distance vectors. This keeps the
    classifier mostly invariant to hand distance from the camera, which is the
    dominant nuisance factor in webcam-only interaction.
    """

    def __init__(self, model_path: str = "models/gesture_svm.pkl", history_len: Optional[int] = None) -> None:
        """Load the trained gesture classifier and its smoothing state.

        Args:
            model_path: Pickle path to the serialized SVM model.
            history_len: Optional override for temporal smoothing length.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the serialized model is missing.
            pickle.UnpicklingError: If the model artifact is invalid.
        """
        cfg = ConfigLoader.get().get("gesture", {})
        history_len = history_len or cfg.get("history_len", 5)
        self.toggle_delta = cfg.get("toggle_delta_threshold", 0.1)

        self.hands = _HandDetector(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(base_dir, model_path)

        # FIXME: Replace pickle serialization with a safer artifact format once
        # the training/export pipeline is formalized.
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.history: collections.deque[int] = collections.deque(maxlen=history_len)
        self.toggle_y_buffer: collections.deque[float] = collections.deque(maxlen=10)

        self.label_map = {0: "SCROLL_UP", 1: "SCROLL_DOWN", 2: None}

        self.last_bounding_box: Optional[Tuple[int, int, int, int]] = None
        self.last_prediction: Optional[str] = None

    def process_frame(self, image: np.ndarray) -> Optional[str]:
        """
        Convert an RGB frame into a stable gesture action.

        Args:
            image: RGB frame with shape ``(height, width, 3)``.

        Returns:
            ``"SCROLL_UP"``, ``"SCROLL_DOWN"``, ``"TOGGLE_MODE"``, or ``None``.

        Raises:
            None.
        """
        h, w, _ = image.shape
        self.last_bounding_box = None
        self.last_prediction = None

        results = self.hands.process(image)
        if not results.multi_hand_landmarks:
            self.history.clear()
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        x_min = min([lm.x for lm in hand_landmarks.landmark])
        x_max = max([lm.x for lm in hand_landmarks.landmark])
        y_min = min([lm.y for lm in hand_landmarks.landmark])
        y_max = max([lm.y for lm in hand_landmarks.landmark])

        self.last_bounding_box = (
            int(x_min * w), int(y_min * h),
            int(x_max * w), int(y_max * h)
        )

        if self.detect_toggle_swipe(hand_landmarks):
            self.history.clear()
            return "TOGGLE_MODE"

        wrist = hand_landmarks.landmark[0]
        features = []
        for i in range(1, 21):
            item = hand_landmarks.landmark[i]
            # Expected shape: (20,) feature vector of wrist-relative distances.
            dist = ((item.x - wrist.x) ** 2 + (item.y - wrist.y) ** 2 + (item.z - wrist.z) ** 2) ** 0.5
            features.append(dist)

        max_val = max(features) if features else 1.0
        if max_val > 0.0:
            features = [f / max_val for f in features]

        X_infer = np.array([features])
        y_pred = self.model.predict(X_infer)[0]

        self.history.append(y_pred)

        if len(self.history) == self.history.maxlen:
            counter = collections.Counter(self.history)
            most_common_label, count = counter.most_common(1)[0]

            if count >= int(self.history.maxlen * 0.8):
                action = self.label_map.get(most_common_label)
                self.last_prediction = action
                return action
            return self.last_prediction

        return None

    def detect_toggle_swipe(self, landmarks) -> bool:
        """Detect the two-finger swipe used to toggle between modes.

        Args:
            landmarks: MediaPipe hand landmarks for the current frame.

        Returns:
            ``True`` when the swipe gesture crosses the configured threshold.

        Raises:
            None.
        """
        lm = landmarks.landmark

        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y

        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y

        if index_extended and middle_extended and ring_folded and pinky_folded:
            avg_y = (lm[8].y + lm[12].y) / 2.0
            self.toggle_y_buffer.append(avg_y)

            if len(self.toggle_y_buffer) == self.toggle_y_buffer.maxlen:
                delta = self.toggle_y_buffer[-1] - self.toggle_y_buffer[0]
                if abs(delta) > self.toggle_delta:
                    self.toggle_y_buffer.clear()
                    return True
        else:
            self.toggle_y_buffer.clear()

        return False
