"""Entry point launching the TorchMoji GUI application."""
from __future__ import annotations

from .gui import TorchMojiApplication


def main() -> int:
    """Launch the GUI and return the Qt exit code."""

    app = TorchMojiApplication()
    return app.run()


if __name__ == "__main__":  # pragma: no cover - convenience execution
    raise SystemExit(main())
