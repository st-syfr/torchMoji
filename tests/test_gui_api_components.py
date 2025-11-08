"""Tests for GUI API settings components.

These tests require a display and Qt platform libraries.
Run with: QT_QPA_PLATFORM=offscreen pytest tests/test_gui_api_components.py
"""
from __future__ import annotations

import os
import sys
import pytest

# Try to import PySide6, skip all tests if it fails
try:
    from PySide6 import QtWidgets
    from torchmoji.gui.app import SettingsDialog, TorchMojiMainWindow
    from torchmoji.settings import TorchMojiSettings
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

pytestmark = pytest.mark.skipif(
    not HAS_GUI,
    reason="GUI tests require PySide6 and Qt platform libraries"
)


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_settings_dialog_includes_api_fields(qapp):
    """Test that settings dialog includes API configuration fields."""
    settings = TorchMojiSettings()
    dialog = SettingsDialog(settings)
    
    # Check that API widgets exist
    assert hasattr(dialog, 'api_enabled_checkbox')
    assert hasattr(dialog, 'api_host_edit')
    assert hasattr(dialog, 'api_port_spin')
    
    # Check initial values
    assert dialog.api_enabled_checkbox.isChecked() is False
    assert dialog.api_host_edit.text() == "127.0.0.1"
    assert dialog.api_port_spin.value() == 5000


def test_settings_dialog_api_values_initialized(qapp):
    """Test that API settings are initialized correctly in dialog."""
    settings = TorchMojiSettings(
        api_enabled=True,
        api_host="0.0.0.0",
        api_port=8080
    )
    dialog = SettingsDialog(settings)
    
    assert dialog.api_enabled_checkbox.isChecked() is True
    assert dialog.api_host_edit.text() == "0.0.0.0"
    assert dialog.api_port_spin.value() == 8080


def test_settings_dialog_get_api_settings(qapp):
    """Test that get_settings returns API configuration."""
    settings = TorchMojiSettings()
    dialog = SettingsDialog(settings)
    
    # Modify API settings in dialog
    dialog.api_enabled_checkbox.setChecked(True)
    dialog.api_host_edit.setText("localhost")
    dialog.api_port_spin.setValue(3000)
    
    # Get settings and verify
    new_settings = dialog.get_settings()
    assert new_settings.api_enabled is True
    assert new_settings.api_host == "localhost"
    assert new_settings.api_port == 3000


def test_settings_dialog_reset_includes_api_defaults(qapp):
    """Test that reset button resets API settings to defaults."""
    settings = TorchMojiSettings(
        api_enabled=True,
        api_host="0.0.0.0",
        api_port=9999
    )
    dialog = SettingsDialog(settings)
    
    # Trigger reset
    dialog._reset_defaults()
    
    # Verify defaults are restored
    assert dialog.api_enabled_checkbox.isChecked() is False
    assert dialog.api_host_edit.text() == "127.0.0.1"
    assert dialog.api_port_spin.value() == 5000


def test_main_window_has_api_status_widgets(qapp):
    """Test that main window includes API status widgets."""
    window = TorchMojiMainWindow()
    
    # Check that API status widgets exist
    assert hasattr(window, 'api_status_label')
    assert hasattr(window, 'api_url_label')
    assert hasattr(window, 'api_info_label')
    
    # Check initial state
    assert "Stopped" in window.api_status_label.text()


def test_main_window_update_api_status(qapp):
    """Test updating API status in main window."""
    window = TorchMojiMainWindow()
    
    # Test running status
    window.update_api_status(True, "127.0.0.1", 5000)
    assert "Running" in window.api_status_label.text()
    assert "127.0.0.1:5000" in window.api_url_label.text()
    
    # Test stopped status
    window.update_api_status(False)
    assert "Stopped" in window.api_status_label.text()
    assert window.api_url_label.text() == ""


def test_api_port_spin_range(qapp):
    """Test that API port spinner has correct range."""
    settings = TorchMojiSettings()
    dialog = SettingsDialog(settings)
    
    # Port should be in valid range (1024-65535)
    assert dialog.api_port_spin.minimum() == 1024
    assert dialog.api_port_spin.maximum() == 65535
