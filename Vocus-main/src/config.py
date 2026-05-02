"""Dynamic configuration parser for Vocus.

The loader keeps the current JSON configuration in memory and reloads it when
the file modification time changes. That makes tuning camera and gesture
parameters possible without restarting the app.
"""

import json
import os
import threading
from typing import Any, Dict


class ConfigLoader:
    """Thread-safe singleton config loader with hot-reload support.

    The class is intentionally simple because it is used on the hot path. The
    configuration file is read only when its modification time changes.
    """
    _lock = threading.Lock()
    _config: Dict[str, Any] = {}
    _last_mtime: float = 0.0

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Return the latest parsed configuration dictionary.

        The loader reads ``config.json`` from the repository root. It updates the
        cached dictionary only when the file timestamp advances, which keeps the
        runtime overhead low while still allowing manual retuning.

        Returns:
            A nested dictionary containing the current configuration values.

        Raises:
            None.
        """
        base_dir = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base_dir, "config.json")

        if not os.path.exists(path):
            return cls._config

        try:
            mtime = os.path.getmtime(path)
            with cls._lock:
                if mtime > cls._last_mtime:
                    with open(path, "r") as f:
                        cls._config = json.load(f)
                    cls._last_mtime = mtime
        except Exception:
            pass  # Fail gracefully if file is locked or malformed

        return cls._config
