# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building the TorchMoji tray application."""
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

project_root = Path(__file__).resolve().parent.parent

package_datas = collect_data_files("torchmoji", includes=["**/*.json"])
extra_datas = [
    (str(project_root / "model"), "model"),
    (str(project_root / "data"), "data"),
]

datas = package_datas + extra_datas

block_cipher = None


a = Analysis(
    [str(project_root / "torchmoji" / "gui_main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="torchmoji-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="torchmoji-gui",
)
