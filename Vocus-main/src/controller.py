"""Translate gesture commands into OS-level scroll actions.

The controller runs on a background thread and consumes the newest command from
the vision pipeline. It intentionally drops stale commands so the desktop reacts
to the latest hand pose rather than replaying a backlog of old intent.
"""

from __future__ import annotations

import platform
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.config import ConfigLoader


# ── Command protocol ────────────────────────────────────────────────


class ScrollDirection(Enum):
    """Enumerate the supported high-level scroll directions.

    The RESET branch is used to map an open-palm gesture to ``Ctrl+0`` on
    Windows and a no-op on platforms that do not expose the shortcut.
    """

    UP = auto()
    DOWN = auto()
    NONE = auto()
    RESET = auto()


@dataclass(frozen=True, slots=True)
class ScrollCommand:
    """Immutable value object representing a single scroll intent.

    Attributes:
        direction: High-level direction or reset action.
        magnitude: Number of scroll clicks to emit.
        continuous: Whether the command should repeat at the fast path.
        zoom: Whether the command should be executed as a zoom shortcut.
    """

    direction: ScrollDirection
    magnitude: int = 1  # scroll "clicks" – tuneable per UX preference
    continuous: bool = False
    zoom: bool = False


# ── Platform-specific scroll backends ───────────────────────────────


def _scroll_windows(cmd: ScrollCommand) -> None:
    """Dispatch a scroll or reset event on Windows.

    Args:
        cmd: Scroll command produced by the vision loop.

    Returns:
        None.

    Raises:
        None.
    """
    import ctypes

    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120
    KEYEVENTF_KEYUP = 0x0002
    VK_CONTROL = 0x11
    VK_0 = 0x30

    if cmd.direction == ScrollDirection.RESET:
        # Ctrl+0 resets zoom in browsers and many desktop applications.
        ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
        try:
            ctypes.windll.user32.keybd_event(VK_0, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_0, 0, KEYEVENTF_KEYUP, 0)
        finally:
            ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
        return

    sign = 1 if cmd.direction == ScrollDirection.UP else -1
    wheel_amount = sign * cmd.magnitude * WHEEL_DELTA

    # ctypes.windll.user32.mouse_event handles unsigned DWORD conversion implicitly.
    if wheel_amount < 0:
        wheel_amount = (wheel_amount + (1 << 32)) & 0xFFFFFFFF

    if cmd.zoom:
        ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
        try:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)
        finally:
            ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    else:
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)


def _scroll_darwin(cmd: ScrollCommand) -> None:
    """Dispatch a scroll event on macOS using Quartz.

    Args:
        cmd: Scroll command produced by the vision loop.

    Returns:
        None.

    Raises:
        None.
    """
    from Quartz import (
        CGEventCreateScrollWheelEvent,
        CGEventPost,
        kCGHIDEventTap,
        kCGScrollEventUnitLine,
    )

    if cmd.direction == ScrollDirection.RESET:
        return

    sign = 1 if cmd.direction == ScrollDirection.UP else -1
    event = CGEventCreateScrollWheelEvent(
        None, kCGScrollEventUnitLine, 1, sign * cmd.magnitude
    )
    CGEventPost(kCGHIDEventTap, event)


def _scroll_linux(cmd: ScrollCommand) -> None:
    """Dispatch a scroll event on Linux via xdotool.

    Args:
        cmd: Scroll command produced by the vision loop.

    Returns:
        None.

    Raises:
        None.
    """
    import subprocess

    if cmd.direction == ScrollDirection.RESET:
        return

    button = "4" if cmd.direction == ScrollDirection.UP else "5"
    for _ in range(cmd.magnitude):
        subprocess.run(
            ["xdotool", "click", button],
            check=False,
            capture_output=True,
        )


# Resolve backend once at import time.
_PLATFORM = platform.system()
_SCROLL_FN = {
    "Windows": _scroll_windows,
    "Darwin": _scroll_darwin,
    "Linux": _scroll_linux,
}.get(_PLATFORM)

if _SCROLL_FN is None:
    raise OSError(f"Unsupported platform: {_PLATFORM}")


# ── Consumer thread ────────────────────────────────────────────────


class ScrollController:
    """Drain a command queue and dispatch OS scroll events.

    The controller deliberately keeps the consumer logic stateful. That lets it
    distinguish pose changes from pose holds, apply separate cooldowns for zoom
    and gesture modes, and keep reset commands immediate.
    """

    def __init__(
        self,
        command_queue: queue.Queue[ScrollCommand],
        cooldown_sec: Optional[float] = None,
    ) -> None:
        """Initialize the queue consumer and resolve runtime cooldowns.

        Args:
            command_queue: Shared queue between the vision producer and this
                consumer.
            cooldown_sec: Optional override for the general scroll cooldown.

        Returns:
            None.

        Raises:
            None.
        """
        cfg = ConfigLoader.get().get("controller", {})
        self._queue = command_queue
        self._cooldown = cooldown_sec if cooldown_sec is not None else cfg.get("scroll_cooldown_sec", 0.1)
        self._gaze_cooldown = float(cfg.get("gaze_scroll_cooldown_sec", self._cooldown))
        self._zoom_cooldown = float(cfg.get("zoom_cooldown_sec", self._gaze_cooldown))
        self._edge_cooldown = float(cfg.get("edge_trigger_cooldown_sec", self._cooldown))
        self._zoom_edge_cooldown = float(cfg.get("zoom_edge_cooldown_sec", 0.0))
        self._gesture_repeat_delay = float(cfg.get("gesture_repeat_delay_sec", 2.5))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._active_pose = ScrollDirection.NONE
        self._active_cmd: Optional[ScrollCommand] = None
        self._pose_start_time = 0.0

    # Context manager mirrors Camera's interface for symmetry.
    def __enter__(self) -> "ScrollController":
        """Start the consumer thread and return the instance.

        Returns:
            The running ``ScrollController`` instance.

        Raises:
            None.
        """
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        """Stop the consumer thread and release queue resources.

        Args:
            *_exc_info: Exception metadata from context manager unwinding.

        Returns:
            None.

        Raises:
            None.
        """
        self.stop()

    def start(self) -> None:
        """Start the background consumer thread.

        Returns:
            None.

        Raises:
            None.
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the consumer thread and unblock any pending queue read.

        Returns:
            None.

        Raises:
            None.
        """
        self._running = False
        # Unblock a potentially waiting .get()
        try:
            self._queue.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _consume_loop(self) -> None:
        """Consume commands and dispatch them with pose-aware rate limiting.

        Returns:
            None.

        Raises:
            None.
        """
        last_action = 0.0
        while self._running:
            try:
                cmd: Optional[ScrollCommand] = self._queue.get(timeout=0.1)
            except queue.Empty:
                # Do not force a NONE pose on a transient queue gap.
                continue

            if cmd is None:
                break  # sentinel received

            current_time = time.monotonic()

            if cmd.direction != self._active_pose:
                # POSE CHANGED
                self._active_pose = cmd.direction
                self._active_cmd = cmd if cmd.direction != ScrollDirection.NONE else None
                self._pose_start_time = current_time

                # TYPEMATIC BEHAVIOR: Execute one scroll instantly upon making the gesture
                if self._active_cmd is not None:
                    if self._active_cmd.direction == ScrollDirection.RESET:
                        _SCROLL_FN(self._active_cmd)
                        last_action = current_time
                    else:
                        edge_cd = self._zoom_edge_cooldown if self._active_cmd.zoom else self._edge_cooldown
                        if current_time - last_action >= edge_cd:
                            _SCROLL_FN(self._active_cmd)
                            last_action = current_time

            else:
                # POSE HELD
                if self._active_pose != ScrollDirection.NONE:
                    # Allow caller to update strength while maintaining same direction.
                    self._active_cmd = cmd
                    elapsed = current_time - self._pose_start_time
                    # Continuous modes repeat immediately, gesture mode keeps hold-to-repeat behavior.
                    repeat_ready = False
                    if self._active_cmd is not None and self._active_cmd.continuous:
                        repeat_ready = True
                    elif elapsed >= self._gesture_repeat_delay:
                        repeat_ready = True

                    if repeat_ready:
                        if self._active_cmd is not None and self._active_cmd.continuous:
                            cd = self._zoom_cooldown if self._active_cmd.zoom else self._gaze_cooldown
                        else:
                            cd = self._cooldown
                        if current_time - last_action >= cd:
                            if self._active_cmd is not None:
                                _SCROLL_FN(self._active_cmd)
                            last_action = current_time
