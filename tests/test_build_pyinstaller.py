"""Tests for the PyInstaller build script."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from textwrap import dedent

import pytest


def test_patch_spec_file_with_empty_lists(tmp_path: Path) -> None:
    """Test patching a spec file with empty datas/binaries lists."""
    from scripts.build_pyinstaller import patch_spec_file, SPEC_FILE, PROJECT_ROOT
    
    # Create a minimal spec file with empty lists
    spec_content = dedent("""\
        # -*- mode: python ; coding: utf-8 -*-
        
        datas = []
        binaries = []
        hiddenimports = []
        
        
        a = Analysis(
            ['gui_main.py'],
            pathex=[],
            binaries=binaries,
            datas=datas,
            hiddenimports=hiddenimports,
        )
    """)
    
    # Mock the SPEC_FILE path to use tmp_path
    original_spec_file = SPEC_FILE
    test_spec_file = tmp_path / "test.spec"
    test_spec_file.write_text(spec_content)
    
    # Monkey-patch the SPEC_FILE constant
    import scripts.build_pyinstaller as build_module
    build_module.SPEC_FILE = test_spec_file
    
    try:
        patch_spec_file()
        
        result = test_spec_file.read_text()
        
        # Check that import was added
        assert "from PyInstaller.utils.hooks import collect_dynamic_libs" in result
        
        # Check that model files were added
        assert 'datas += [' in result
        assert '("model/vocabulary.json", "model")' in result
        assert '("model/pytorch_model.bin", "model")' in result
        
        # Check that torch binaries were added
        assert 'binaries += collect_dynamic_libs("torch")' in result
        
        # Check that original empty lists are still there
        assert 'datas = []' in result
        assert 'binaries = []' in result
        
    finally:
        # Restore the original SPEC_FILE
        build_module.SPEC_FILE = original_spec_file


def test_patch_spec_file_with_collect_all(tmp_path: Path) -> None:
    """Test patching a spec file that already has collect_all entries."""
    from scripts.build_pyinstaller import patch_spec_file, SPEC_FILE
    
    # Create a spec file with collect_all PySide6
    spec_content = dedent("""\
        # -*- mode: python ; coding: utf-8 -*-
        from PyInstaller.utils.hooks import collect_all
        
        datas = []
        binaries = []
        hiddenimports = []
        tmp_ret = collect_all('PySide6')
        datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
        
        
        a = Analysis(
            ['gui_main.py'],
            pathex=[],
            binaries=binaries,
            datas=datas,
            hiddenimports=hiddenimports,
        )
    """)
    
    test_spec_file = tmp_path / "test.spec"
    test_spec_file.write_text(spec_content)
    
    # Monkey-patch the SPEC_FILE constant
    import scripts.build_pyinstaller as build_module
    original_spec_file = build_module.SPEC_FILE
    build_module.SPEC_FILE = test_spec_file
    
    try:
        patch_spec_file()
        
        result = test_spec_file.read_text()
        
        # Check that import was added (collect_dynamic_libs)
        assert "from PyInstaller.utils.hooks import collect_dynamic_libs" in result
        
        # Check that model files were added
        assert 'datas += [' in result
        assert '("model/vocabulary.json", "model")' in result
        assert '("model/pytorch_model.bin", "model")' in result
        
        # Check that torch binaries were added
        assert 'binaries += collect_dynamic_libs("torch")' in result
        
        # Check that PySide6 collect_all is still there
        assert "collect_all('PySide6')" in result
        assert 'datas += tmp_ret[0]' in result
        
    finally:
        build_module.SPEC_FILE = original_spec_file


def test_patch_spec_file_idempotent(tmp_path: Path) -> None:
    """Test that patching the same file twice doesn't duplicate entries."""
    from scripts.build_pyinstaller import patch_spec_file, SPEC_FILE
    
    spec_content = dedent("""\
        # -*- mode: python ; coding: utf-8 -*-
        
        datas = []
        binaries = []
        hiddenimports = []
        
        
        a = Analysis(
            ['gui_main.py'],
            pathex=[],
            binaries=binaries,
            datas=datas,
            hiddenimports=hiddenimports,
        )
    """)
    
    test_spec_file = tmp_path / "test.spec"
    test_spec_file.write_text(spec_content)
    
    import scripts.build_pyinstaller as build_module
    original_spec_file = build_module.SPEC_FILE
    build_module.SPEC_FILE = test_spec_file
    
    try:
        # Patch twice
        patch_spec_file()
        first_result = test_spec_file.read_text()
        
        patch_spec_file()
        second_result = test_spec_file.read_text()
        
        # Results should be identical
        assert first_result == second_result
        
        # Should only have one instance of each addition
        assert first_result.count('("model/vocabulary.json", "model")') == 1
        assert first_result.count('collect_dynamic_libs("torch")') == 1
        
    finally:
        build_module.SPEC_FILE = original_spec_file


def test_clean_build_artifacts_handles_permission_error(tmp_path: Path, monkeypatch) -> None:
    """Test that clean_build_artifacts provides helpful error message on PermissionError."""
    from scripts.build_pyinstaller import clean_build_artifacts, PROJECT_ROOT
    import scripts.build_pyinstaller as build_module
    
    # Create a test directory structure
    test_root = tmp_path / "test_project"
    test_root.mkdir()
    dist_dir = test_root / "dist"
    dist_dir.mkdir()
    test_exe = dist_dir / "test.exe"
    test_exe.write_text("fake exe")
    
    # Monkey-patch PROJECT_ROOT
    original_root = build_module.PROJECT_ROOT
    build_module.PROJECT_ROOT = test_root
    
    # Create a mock that simulates permission error on first attempts, then succeeds
    original_rmtree = shutil.rmtree
    attempt_count = [0]
    
    def mock_rmtree_fail_once(path, *args, **kwargs):
        attempt_count[0] += 1
        if attempt_count[0] <= 1:
            raise PermissionError(f"Access denied: {path}")
        # On retry, succeed
        original_rmtree(path, *args, **kwargs)
    
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree_fail_once)
    
    try:
        # Should succeed after retry
        clean_build_artifacts(include_dist=True, include_spec=False)
        
        # Verify directory was removed
        assert not dist_dir.exists()
        # Verify retry was attempted
        assert attempt_count[0] == 2
        
    finally:
        build_module.PROJECT_ROOT = original_root


def test_clean_build_artifacts_fails_with_helpful_message(tmp_path: Path, monkeypatch) -> None:
    """Test that clean_build_artifacts gives helpful error message after max retries."""
    from scripts.build_pyinstaller import clean_build_artifacts, PROJECT_ROOT
    import scripts.build_pyinstaller as build_module
    
    # Create a test directory
    test_root = tmp_path / "test_project"
    test_root.mkdir()
    dist_dir = test_root / "dist"
    dist_dir.mkdir()
    
    # Monkey-patch PROJECT_ROOT
    original_root = build_module.PROJECT_ROOT
    build_module.PROJECT_ROOT = test_root
    
    # Mock rmtree to always fail
    def mock_rmtree_always_fail(path, *args, **kwargs):
        raise PermissionError(f"Access denied: {path}")
    
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree_always_fail)
    
    try:
        # Should raise PermissionError with helpful message
        with pytest.raises(PermissionError) as exc_info:
            clean_build_artifacts(include_dist=True, include_spec=False)
        
        error_message = str(exc_info.value)
        # Check that helpful error message is present
        assert "Common causes on Windows:" in error_message
        assert "The executable" in error_message
        assert "Windows Defender" in error_message
        assert "Try the following:" in error_message
        
    finally:
        build_module.PROJECT_ROOT = original_root
