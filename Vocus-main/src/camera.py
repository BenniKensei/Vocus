"""Thread-safe OpenCV video capture wrapper.

This module isolates camera I/O from the main inference loop so frame reads do
not block gesture inference. The reader thread keeps the most recent frame in a
shared buffer, and consumers take a copy to avoid race conditions.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    """Continuously grab frames from a video device in a background thread.

    The class exists to reduce end-to-end latency. Camera reads are I/O-bound,
    while gesture inference is CPU-bound, so separating them improves frame
    cadence and keeps the UI responsive.

    Attributes:
        _cap: OpenCV capture handle.
        _lock: Guards access to the shared frame buffer.
        _frame: Latest captured frame.
        _running: Indicates whether the reader thread should keep looping.
        _thread: Background thread that pulls frames from the device.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        api_preference: int = cv2.CAP_ANY,
    ) -> None:
        """Initialize the capture device and request the target resolution.

        Args:
            device_index: Camera index passed to OpenCV.
            width: Requested capture width in pixels.
            height: Requested capture height in pixels.
            api_preference: OpenCV backend preference.

        Returns:
            None.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(device_index, api_preference)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Camera device {device_index} could not be opened. "
                "Check permissions and hardware connection."
            )

        # Request resolution – the driver may clamp to the nearest supported size.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        """Start the camera reader and return the instance.

        Returns:
            The running ``Camera`` instance.

        Raises:
            None.
        """
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        """Stop the camera reader and release the device.

        Args:
            *_exc_info: Exception metadata from the context manager protocol.

        Returns:
            None.

        Raises:
            None.
        """
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the background reader thread.

        Returns:
            None.

        Raises:
            None.
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the reader thread to exit and release the capture device.

        Returns:
            None.

        Raises:
            None.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recent frame copy.

        Returns:
            A tuple of ``(success, frame)`` matching the OpenCV contract.
            ``frame`` is copied so consumers cannot mutate the shared buffer.

        Raises:
            None.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return the actual resolution negotiated with the driver.

        Returns:
            A ``(width, height)`` tuple in pixels.

        Raises:
            None.
        """
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Continuously grab frames until ``stop()`` is called.

        Returns:
            None.

        Raises:
            None.
        """
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue  # transient failure – retry on next iteration
            with self._lock:
                self._frame = frame
