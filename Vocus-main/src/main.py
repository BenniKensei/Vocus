"""Application entry point for the Vocus producer-consumer pipeline.

The runtime separates frame acquisition, gesture inference, and OS dispatch so
camera I/O does not block command emission.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
import tkinter as tk

import cv2

from src.camera import Camera
from src.config import ConfigLoader
from src.controller import ScrollCommand, ScrollController, ScrollDirection
from src.tracker import GestureTracker, ZoomTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vocus")

_quit_event = threading.Event()
global_preview = False
global_active_mode = "ZOOM"

def parse_args() -> argparse.Namespace:
    """Parse the command-line interface for the camera pipeline.

    Returns:
        Parsed CLI arguments.

    Raises:
        None.
    """
    p = argparse.ArgumentParser(description="Vocus – zoom & gesture interaction controller")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cooldown", type=float, default=None)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--debug", action="store_true", help="Force preview on via CLI")
    return p.parse_args()

def _vision_loop(args: argparse.Namespace) -> None:
    """Run the frame-processing loop and enqueue OS commands.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None.

    Raises:
        None.
    """
    global global_active_mode
    cmd_queue: queue.Queue[ScrollCommand] = queue.Queue(maxsize=1)
    zoom_tracker = ZoomTracker()
    gesture_tracker = GestureTracker()
    
    with Camera(args.device, args.width, args.height) as cam, \
         ScrollController(cmd_queue, cooldown_sec=args.cooldown):

        actual_w, actual_h = cam.resolution
        log.info("Pipeline started  |  resolution %dx%d", actual_w, actual_h)

        fps_timer = time.monotonic()
        frame_count = 0
        
        was_preview = False

        while not _quit_event.is_set():
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            # In headless mode (no preview active), throttle CPU
            if not global_preview:
                time.sleep(0.01)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            zoom_action = None
            gesture_action = None
            if global_active_mode == "ZOOM":
                zoom_action = zoom_tracker.process_frame(rgb_frame)
            else:
                gesture_action = gesture_tracker.process_frame(rgb_frame)
            
            if gesture_action == "TOGGLE_MODE":
                global_active_mode = "GESTURE" if global_active_mode == "ZOOM" else "ZOOM"
                log.info("Mode manually toggled -> %s", global_active_mode)
                # Let the UI know if it's polling, but it's simpler to just update the button later if clicked
                time.sleep(1.0)
                continue
                
            cfg_controller = ConfigLoader.get().get("controller", {})
            mag = cfg_controller.get("magnitude", 1)

            action_str = None
            action_mag = mag
            action_continuous = False
            action_zoom = False
            if global_active_mode == "ZOOM":
                if zoom_action is not None:
                    action_str, action_mag = zoom_action
                    action_continuous = True
                    action_zoom = True
            elif global_active_mode == "GESTURE":
                action_str = gesture_action
            
            # Enqueue NONE commands so the controller can reset its pose state.
            if action_str == "SCROLL_UP":
                cmd = ScrollCommand(
                    direction=ScrollDirection.UP,
                    magnitude=action_mag,
                    continuous=action_continuous,
                    zoom=action_zoom,
                )
            elif action_str == "SCROLL_DOWN":
                cmd = ScrollCommand(
                    direction=ScrollDirection.DOWN,
                    magnitude=action_mag,
                    continuous=action_continuous,
                    zoom=action_zoom,
                )
            elif action_str == "ZOOM_IN":
                cmd = ScrollCommand(
                    direction=ScrollDirection.UP,
                    magnitude=action_mag,
                    continuous=True,
                    zoom=True,
                )
            elif action_str == "ZOOM_OUT":
                cmd = ScrollCommand(
                    direction=ScrollDirection.DOWN,
                    magnitude=action_mag,
                    continuous=True,
                    zoom=True,
                )
            elif action_str == "RESET_ZOOM":
                cmd = ScrollCommand(
                    direction=ScrollDirection.RESET,
                    magnitude=1,
                    continuous=False,
                    zoom=True,
                )
            else:
                cmd = ScrollCommand(direction=ScrollDirection.NONE, magnitude=mag)
                
            try:
                cmd_queue.put_nowait(cmd)
            except queue.Full:
                pass

            if global_preview:
                was_preview = True
                frame_count += 1
                elapsed = time.monotonic() - fps_timer
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_timer = time.monotonic()
                else:
                    fps = frame_count / max(elapsed, 1e-9)

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"MODE: {global_active_mode}", (actual_w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if global_active_mode == "GESTURE" else (0, 255, 255), 2)
                if global_active_mode == "ZOOM":
                    zoom_state = "TRACKING" if zoom_tracker.last_metric is not None else "SEARCHING"
                    cv2.putText(
                        frame,
                        f"ZOOM: {zoom_state}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if zoom_state == "TRACKING" else (0, 165, 255),
                        2,
                    )
                    if zoom_tracker.last_metric is not None:
                        cv2.putText(
                            frame,
                            f"OPENNESS: {zoom_tracker.last_metric:.2f}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )
                    if zoom_tracker.last_bounding_box:
                        x1, y1, x2, y2 = zoom_tracker.last_bounding_box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                else:
                    if gesture_tracker.last_bounding_box:
                        x1, y1, x2, y2 = gesture_tracker.last_bounding_box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    if gesture_tracker.last_prediction:
                        cv2.putText(frame, f"Gesture: {gesture_tracker.last_prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                cv2.imshow("Vocus Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit signal received via Preview UI.")
                    _quit_event.set()
                    break
            else:
                # Clean up preview window immediately when toggled off
                if was_preview:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)  # allow destruction to flush
                    was_preview = False
                    
        # Explicitly clean up any stuck windows bound to this thread
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def run_ui():
    """Run the small Tkinter control panel for preview and mode toggles.

    Returns:
        None.

    Raises:
        None.
    """
    root = tk.Tk()
    root.title("Vocus Control")
    root.geometry("220x160")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')

    def toggle_preview():
        global global_preview
        global_preview = not global_preview
        btn_preview.config(text="Hide Preview" if global_preview else "Show Preview")

    def toggle_mode():
        global global_active_mode
        global_active_mode = "GESTURE" if global_active_mode == "ZOOM" else "ZOOM"
        btn_mode.config(text=f"Mode: {global_active_mode}")

    def quit_app():
        _quit_event.set()
        root.destroy()

    frame = tk.Frame(root, padx=20, pady=15)
    frame.pack(expand=True, fill=tk.BOTH)

    btn_preview = tk.Button(frame, text="Hide Preview" if global_preview else "Show Preview", command=toggle_preview, width=15)
    btn_preview.pack(pady=5)
    
    btn_mode = tk.Button(frame, text=f"Mode: {global_active_mode}", command=toggle_mode, width=15)
    btn_mode.pack(pady=5)
    
    btn_quit = tk.Button(frame, text="Quit Pipeline", command=quit_app, width=15, fg="red")
    btn_quit.pack(pady=5)

    def check_quit():
        if _quit_event.is_set():
            root.destroy()
        else:
            root.after(200, check_quit)

    check_quit()
    root.mainloop()


def main() -> None:
    """Start the background vision thread and the control UI.

    Returns:
        None.

    Raises:
        None.
    """
    global global_preview, global_active_mode
    args = parse_args()
    cfg = ConfigLoader.get().get("app", {})
    
    # Initialize preview state from arguments
    global_preview = args.debug or (cfg.get("preview", False) and not args.no_preview)
    global_active_mode = cfg.get("default_mode", "ZOOM")
    
    log.info("Booting vision daemon in background mode.")
    t = threading.Thread(target=_vision_loop, args=(args,), daemon=True)
    t.start()
    
    log.info("Starting Tkinter Control UI.")
    run_ui()
        
    log.info("Waiting for pipeline thread to terminate gracefully...")
    t.join(timeout=3.0)
    
    log.info("Pipeline cleanly terminated.")

if __name__ == "__main__":
    main()
