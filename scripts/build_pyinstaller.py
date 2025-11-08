"""Utility to build the TorchMoji GUI with PyInstaller.

This automates the manual steps documented in the repository by:
* ensuring the pretrained model assets are available;
* generating the PyInstaller spec file for the GUI application; and
* patching the spec so the frozen binary bundles the model data and Torch
  dynamic libraries before producing the final executable.

Usage::

    python scripts/build_pyinstaller.py

The script assumes PyInstaller is installed in the active environment. Pass
``--clean`` to remove previous build artifacts before running, and
``--skip-download`` to skip fetching the model weights if they are already in
place.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRY_SCRIPT = PROJECT_ROOT / "torchmoji" / "gui_main.py"
SPEC_FILE = PROJECT_ROOT / "torchmoji-gui.spec"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILES = (
    MODEL_DIR / "vocabulary.json",
    MODEL_DIR / "pytorch_model.bin",
)


def run_command(cmd: Iterable[str], *, cwd: Path | None = None) -> None:
    """Run *cmd* in a subprocess, streaming output and raising on failure."""
    display_cmd = " ".join(cmd)
    print(f"[build_pyinstaller] Running: {display_cmd}")
    subprocess.run(cmd, check=True, cwd=cwd or PROJECT_ROOT)


def ensure_pyinstaller_available() -> None:
    try:
        import PyInstaller  # noqa: F401  # Import to validate availability.
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "PyInstaller is not installed. Install it with 'pip install PyInstaller' "
            "before running this script."
        ) from exc


def ensure_model_assets(*, skip_download: bool) -> None:
    missing = [path for path in MODEL_FILES if not path.exists()]
    if not missing:
        return

    if skip_download:
        missing_list = ", ".join(path.name for path in missing)
        raise SystemExit(
            f"Missing model asset(s): {missing_list}. Rerun without --skip-download "
            "or fetch them manually."
        )

    print("[build_pyinstaller] Downloading TorchMoji model assets…")
    run_command([sys.executable, "scripts/download_weights.py"])
    still_missing = [path for path in MODEL_FILES if not path.exists()]
    if still_missing:
        missing_list = ", ".join(path.name for path in still_missing)
        raise SystemExit(
            f"Model download script completed but the following files are still missing:"
            f" {missing_list}"
        )


def ensure_entry_script() -> None:
    """Verify the GUI entry script exists."""
    if not ENTRY_SCRIPT.exists():
        raise SystemExit(
            f"GUI entry script not found at {ENTRY_SCRIPT}. "
            "Ensure the torchmoji package is properly installed."
        )


def generate_spec_file() -> None:
    if SPEC_FILE.exists():
        print("[build_pyinstaller] Removing existing spec file to regenerate a fresh copy.")
        SPEC_FILE.unlink()
    run_command(
        [
            sys.executable,
            "-m",
            "PyInstaller.utils.cliutils.makespec",
            str(ENTRY_SCRIPT),
            "--onefile",
            "--name",
            "torchmoji-gui",
            "--noconsole",
            "--collect-submodules",
            "torch",
            "--collect-submodules",
            "numpy",
            "--collect-submodules",
            "sklearn",
            "--collect-all",
            "emoji",
            "--collect-all",
            "PySide6",
        ]
    )


def patch_spec_file() -> None:
    if not SPEC_FILE.exists():
        raise SystemExit("Spec file not found; run PyInstaller generation step first.")

    text = SPEC_FILE.read_text()
    changed = False

    if "collect_dynamic_libs" not in text:
        text = "from PyInstaller.utils.hooks import collect_dynamic_libs\n" + text
        changed = True

    # Add model files to datas list by appending after existing entries
    if "model/vocabulary.json" not in text:
        # Find where to insert model datas - look for the line before Analysis(
        analysis_pattern = re.compile(r"\n(a = Analysis\()")
        match = analysis_pattern.search(text)
        if not match:
            raise SystemExit("Failed to locate Analysis section in spec file.")
        
        datas_addition = (
            "datas += [\n"
            "    (\"model/vocabulary.json\", \"model\"),\n"
            "    (\"model/pytorch_model.bin\", \"model\"),\n"
            "]\n"
        )
        # Insert before the Analysis line
        insert_pos = match.start()
        text = text[:insert_pos] + "\n" + datas_addition + text[insert_pos:]
        changed = True

    # Add torch dynamic libs to binaries list by appending after existing entries
    if "collect_dynamic_libs(\"torch\")" not in text:
        # Find where to insert torch binaries - look for the line before Analysis(
        analysis_pattern = re.compile(r"\n(a = Analysis\()")
        match = analysis_pattern.search(text)
        if not match:
            raise SystemExit("Failed to locate Analysis section in spec file.")
        
        binaries_addition = 'binaries += collect_dynamic_libs("torch")\n'
        # Insert before the Analysis line
        insert_pos = match.start()
        text = text[:insert_pos] + "\n" + binaries_addition + text[insert_pos:]
        changed = True

    if changed:
        SPEC_FILE.write_text(text)
        print("[build_pyinstaller] Updated torchmoji-gui.spec with model data and Torch binaries.")
    else:
        print("[build_pyinstaller] Spec file already contains the required patches.")


def run_spec_build() -> None:
    run_command([sys.executable, "-m", "PyInstaller", str(SPEC_FILE)])


def clean_build_artifacts(
    *, include_dist: bool = True, include_spec: bool = False, context: str | None = None
) -> None:
    """Remove cached files from previous PyInstaller builds."""

    def describe(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    message_context = f" ({context})" if context else ""

    targets: list[Path] = []

    if include_spec and SPEC_FILE.exists():
        targets.append(SPEC_FILE)

    for name in ("build", "__pycache__"):
        path = PROJECT_ROOT / name
        if path.exists():
            targets.append(path)

    if include_dist:
        dist_path = PROJECT_ROOT / "dist"
        if dist_path.exists():
            targets.append(dist_path)

    for path in targets:
        print(
            f"[build_pyinstaller] Removing cached artifact{message_context}: {describe(path)}."
        )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous build artifacts before running PyInstaller.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume model files are present and skip running download_weights.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ensure_pyinstaller_available()

    clean_build_artifacts(
        include_dist=args.clean, include_spec=True, context="pre-build"
    )

    ensure_model_assets(skip_download=args.skip_download)
    ensure_entry_script()
    generate_spec_file()
    patch_spec_file()
    run_spec_build()

    print("[build_pyinstaller] Build complete. Check the dist/ directory for the GUI executable.")

    clean_build_artifacts(include_dist=False, include_spec=True, context="post-build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
