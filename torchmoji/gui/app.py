"""PySide6-based system tray application for TorchMoji."""
from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import emoji
from PySide6 import QtCore, QtGui, QtWidgets

from ..cli import EMOTION_BUNDLES
from ..emojis import EMOJI_ALIASES
from ..runtime import EmojiPredictionResult, EmojiSelection, get_runtime
from ..settings import TorchMojiSettings, load_settings, save_settings
from .utils import build_cli_command, format_cli_command, resolve_prediction_settings

__all__ = ["TorchMojiApplication"]


class WorkerSignals(QtCore.QObject):
    """Signals emitted by :class:`PredictionWorker`."""

    finished = QtCore.Signal(str, object)
    failed = QtCore.Signal(str)
    completed = QtCore.Signal(object)


class PredictionWorker(QtCore.QRunnable):
    """Background task that executes a model prediction."""

    def __init__(self, text: str, settings: TorchMojiSettings) -> None:
        super().__init__()
        self.text = text
        self.settings = replace(settings)
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:  # pragma: no cover - exercised via GUI runtime
        try:
            runtime = get_runtime(Path(self.settings.weights), Path(self.settings.vocab), self.settings.maxlen)
            prediction_settings = resolve_prediction_settings(self.settings)
            result = runtime.predict(self.text, prediction_settings)
        except Exception as exc:  # pragma: no cover - surfaced through GUI
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(self.text, result)
        finally:
            self.signals.completed.emit(self)


class SettingsDialog(QtWidgets.QDialog):
    """Dialog allowing users to configure ``TorchMojiSettings`` values."""

    def __init__(self, settings: TorchMojiSettings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TorchMoji Settings")
        self._settings = replace(settings)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.top_k_spin = QtWidgets.QSpinBox()
        self.top_k_spin.setRange(1, 64)
        self.top_k_spin.setValue(self._settings.top_k)
        form.addRow("Top K", self.top_k_spin)

        self.maxlen_spin = QtWidgets.QSpinBox()
        self.maxlen_spin.setRange(1, 512)
        self.maxlen_spin.setValue(self._settings.maxlen)
        form.addRow("Max length", self.maxlen_spin)

        self.scores_checkbox = QtWidgets.QCheckBox("Display prediction scores")
        self.scores_checkbox.setChecked(self._settings.scores)
        form.addRow("Scores", self.scores_checkbox)

        self.mode_combo = QtWidgets.QComboBox()
        for key in sorted(EMOTION_BUNDLES.keys()):
            self.mode_combo.addItem(key)
        index = self.mode_combo.findText(self._settings.mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        form.addRow("Emotion bundle", self.mode_combo)

        self.emotions_edit = QtWidgets.QLineEdit(self._join_list(self._settings.emotions))
        self.emotions_edit.setPlaceholderText("Comma separated override; leave empty for defaults")
        form.addRow("Emotions", self.emotions_edit)

        self.weak_edit = QtWidgets.QLineEdit(self._join_list(self._settings.weak_emotions))
        self.weak_edit.setPlaceholderText("Comma separated weak emotions; leave empty for defaults")
        form.addRow("Weak emotions", self.weak_edit)

        self.strong_edit = QtWidgets.QLineEdit(self._join_list(self._settings.strong_emotions))
        self.strong_edit.setPlaceholderText("Comma separated strong emotions; leave empty for defaults")
        form.addRow("Strong emotions", self.strong_edit)

        self.weights_edit = QtWidgets.QLineEdit(self._settings.weights)
        self.weights_button = QtWidgets.QPushButton("Browse…")
        weights_layout = QtWidgets.QHBoxLayout()
        weights_layout.addWidget(self.weights_edit)
        weights_layout.addWidget(self.weights_button)
        form.addRow("Weights", weights_layout)

        self.vocab_edit = QtWidgets.QLineEdit(self._settings.vocab)
        self.vocab_button = QtWidgets.QPushButton("Browse…")
        vocab_layout = QtWidgets.QHBoxLayout()
        vocab_layout.addWidget(self.vocab_edit)
        vocab_layout.addWidget(self.vocab_button)
        form.addRow("Vocab", vocab_layout)

        # API Server Settings Section
        api_group = QtWidgets.QGroupBox("Local API Server")
        api_layout = QtWidgets.QFormLayout(api_group)
        
        self.api_enabled_checkbox = QtWidgets.QCheckBox("Enable API server on startup")
        self.api_enabled_checkbox.setChecked(self._settings.api_enabled)
        api_layout.addRow("Enable", self.api_enabled_checkbox)
        
        self.api_host_edit = QtWidgets.QLineEdit(self._settings.api_host)
        self.api_host_edit.setPlaceholderText("e.g., 127.0.0.1 or 0.0.0.0")
        api_layout.addRow("Host", self.api_host_edit)
        
        self.api_port_spin = QtWidgets.QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(self._settings.api_port)
        api_layout.addRow("Port", self.api_port_spin)
        
        # Add informational text
        info_label = QtWidgets.QLabel(
            "The local API server allows other applications to interact with TorchMoji "
            "while the GUI is running. When enabled, you can send prediction requests "
            "via HTTP. See README for API documentation."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: gray; font-size: 9pt; margin-top: 5px; }")
        api_layout.addRow(info_label)
        
        layout.addWidget(api_group)

        self.weights_button.clicked.connect(lambda: self._browse_path(self.weights_edit))
        self.vocab_button.clicked.connect(lambda: self._browse_path(self.vocab_edit))

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.reset_button = button_box.addButton("Reset to defaults", QtWidgets.QDialogButtonBox.ActionRole)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.reset_button.clicked.connect(self._reset_defaults)

    @staticmethod
    def _join_list(values: Iterable[str] | None) -> str:
        if not values:
            return ""
        return ", ".join(values)

    def _browse_path(self, line_edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", line_edit.text() or str(Path.home()))
        if path:
            line_edit.setText(path)

    def _reset_defaults(self) -> None:
        defaults = TorchMojiSettings()
        self.top_k_spin.setValue(defaults.top_k)
        self.maxlen_spin.setValue(defaults.maxlen)
        self.scores_checkbox.setChecked(defaults.scores)
        index = self.mode_combo.findText(defaults.mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        self.emotions_edit.clear()
        self.weak_edit.clear()
        self.strong_edit.clear()
        self.weights_edit.setText(defaults.weights)
        self.vocab_edit.setText(defaults.vocab)
        self.api_enabled_checkbox.setChecked(defaults.api_enabled)
        self.api_host_edit.setText(defaults.api_host)
        self.api_port_spin.setValue(defaults.api_port)

    def _parse_list(self, text: str) -> list[str] | None:
        items = [item.strip() for item in text.replace("\n", ",").split(",")]
        filtered = [item for item in items if item]
        return filtered or None

    def get_settings(self) -> TorchMojiSettings:
        return TorchMojiSettings(
            top_k=self.top_k_spin.value(),
            maxlen=self.maxlen_spin.value(),
            scores=self.scores_checkbox.isChecked(),
            mode=self.mode_combo.currentText(),
            emotions=self._parse_list(self.emotions_edit.text()),
            weak_emotions=self._parse_list(self.weak_edit.text()),
            strong_emotions=self._parse_list(self.strong_edit.text()),
            weights=self.weights_edit.text(),
            vocab=self.vocab_edit.text(),
            api_enabled=self.api_enabled_checkbox.isChecked(),
            api_host=self.api_host_edit.text(),
            api_port=self.api_port_spin.value(),
        )


class TorchMojiMainWindow(QtWidgets.QWidget):
    """Main widget hosting the text editor and results view."""

    settings_requested = QtCore.Signal()
    text_changed = QtCore.Signal(str)
    hidden_to_tray = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TorchMoji")
        self.resize(760, 640)
        self._allow_close = False
        self._command_tokens: list[str] = []

        layout = QtWidgets.QVBoxLayout(self)

        header_layout = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel("Input text")
        header_layout.addWidget(header_label)
        header_layout.addStretch(1)
        self.settings_button = QtWidgets.QPushButton("Settings…")
        header_layout.addWidget(self.settings_button)
        layout.addLayout(header_layout)

        self.text_input = QtWidgets.QPlainTextEdit()
        self.text_input.setPlaceholderText("Type text to emojize…")
        layout.addWidget(self.text_input)

        layout.addWidget(QtWidgets.QLabel("Predicted emojis"))
        self.results_view = QtWidgets.QTreeWidget()
        self.results_view.setColumnCount(3)
        self.results_view.setHeaderLabels(["Emoji", "Alias", "Score"])
        self.results_view.setRootIsDecorated(False)
        self.results_view.setUniformRowHeights(True)
        layout.addWidget(self.results_view, 1)

        self.status_label = QtWidgets.QLabel("Enter text to see predictions.")
        layout.addWidget(self.status_label)

        command_box = QtWidgets.QGroupBox("CLI command preview")
        command_layout = QtWidgets.QVBoxLayout(command_box)
        self.command_label = QtWidgets.QLabel()
        self.command_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        command_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.command_label.setFont(command_font)
        self.command_label.setWordWrap(True)
        command_layout.addWidget(self.command_label)
        self.copy_button = QtWidgets.QPushButton("Copy to clipboard")
        command_layout.addWidget(self.copy_button, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(command_box)

        # API Server Status Section
        api_box = QtWidgets.QGroupBox("Local API Server")
        api_layout = QtWidgets.QVBoxLayout(api_box)
        
        status_layout = QtWidgets.QHBoxLayout()
        self.api_status_label = QtWidgets.QLabel("Status: Stopped")
        status_layout.addWidget(self.api_status_label)
        status_layout.addStretch(1)
        api_layout.addLayout(status_layout)
        
        self.api_url_label = QtWidgets.QLabel()
        self.api_url_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        api_url_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.api_url_label.setFont(api_url_font)
        api_layout.addWidget(self.api_url_label)
        
        self.api_info_label = QtWidgets.QLabel(
            "Enable the API server in Settings to allow external applications to interact with TorchMoji. "
            "See README for endpoint documentation."
        )
        self.api_info_label.setWordWrap(True)
        self.api_info_label.setStyleSheet("QLabel { color: gray; font-size: 9pt; }")
        api_layout.addWidget(self.api_info_label)
        
        layout.addWidget(api_box)

        self.settings_button.clicked.connect(self.settings_requested.emit)
        self.copy_button.clicked.connect(self._copy_command)
        self.text_input.textChanged.connect(self._emit_text_changed)

    def _emit_text_changed(self) -> None:
        self.text_changed.emit(self.text_input.toPlainText())

    def _copy_command(self) -> None:
        if not self._command_tokens:
            return
        QtGui.QGuiApplication.clipboard().setText(format_cli_command(self._command_tokens))

    def set_cli_command_tokens(self, tokens: Sequence[str]) -> None:
        self._command_tokens = list(tokens)
        if tokens:
            self.command_label.setText(format_cli_command(tokens))
        else:
            self.command_label.clear()

    def clear_results(self) -> None:
        self.results_view.clear()
        self.status_label.setText("Enter text to see predictions.")

    def show_predictions(self, result: EmojiPredictionResult, show_scores: bool) -> None:
        self.results_view.clear()
        for selection in result.selections:
            alias = self._alias_for_selection(selection)
            emoji_char = self._emojize(alias)
            score = f"{selection.score:.4f}" if show_scores and selection.score is not None else ""
            item = QtWidgets.QTreeWidgetItem([emoji_char, alias, score])
            self.results_view.addTopLevelItem(item)
        if result.selections:
            self.status_label.setText(f"Showing {len(result.selections)} predictions.")
        else:
            self.status_label.setText("No predictions available.")

    def show_error(self, message: str) -> None:
        self.status_label.setText(message)
        QtWidgets.QMessageBox.warning(self, "TorchMoji", message)

    def update_api_status(self, running: bool, host: str = "", port: int = 0) -> None:
        """Update the API server status display."""
        if running:
            self.api_status_label.setText(f"Status: Running")
            self.api_url_label.setText(f"API URL: http://{host}:{port}")
        else:
            self.api_status_label.setText("Status: Stopped")
            self.api_url_label.setText("")

    def show_main(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def prepare_to_close(self) -> None:
        self._allow_close = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI hook
        if self._allow_close:
            super().closeEvent(event)
            return
        event.ignore()
        self.hide()
        self.hidden_to_tray.emit()

    @staticmethod
    def _alias_for_selection(selection: EmojiSelection) -> str:
        try:
            return EMOJI_ALIASES[selection.index]
        except (IndexError, KeyError):  # pragma: no cover - defensive
            return str(selection.index)

    @staticmethod
    def _emojize(alias: str) -> str:
        try:
            return emoji.emojize(alias, language="alias")
        except TypeError:  # pragma: no cover - emoji<2 fallback
            return emoji.emojize(alias, use_aliases=True)


class TorchMojiApplication(QtCore.QObject):
    """Bootstrapper wiring together the GUI widgets and runtime."""

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__()
        qt_args = list(argv) if argv is not None else sys.argv
        self._qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(qt_args)
        self._settings = load_settings()
        self._current_text = ""
        self._prediction_timer = QtCore.QTimer()
        self._prediction_timer.setInterval(300)
        self._prediction_timer.setSingleShot(True)
        self._prediction_timer.timeout.connect(self._run_prediction)
        self._thread_pool = QtCore.QThreadPool()
        self._active_workers: set[PredictionWorker] = set()
        self._tray_notified = False
        self._api_server = None
        self._api_thread = None

        self._main_window = TorchMojiMainWindow()
        self._main_window.settings_requested.connect(self._open_settings_dialog)
        self._main_window.text_changed.connect(self._on_text_changed)
        self._main_window.hidden_to_tray.connect(self._maybe_notify_tray)

        self._tray_icon = self._create_tray_icon()
        self._update_cli_preview()
        
        # Start API server if enabled in settings
        if self._settings.api_enabled:
            self._start_api_server()
        else:
            self._main_window.update_api_status(False)
        
        self._main_window.show()

    def _create_tray_icon(self) -> QtWidgets.QSystemTrayIcon:
        icon = self._qt_app.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        tray = QtWidgets.QSystemTrayIcon(icon, self._main_window)
        menu = QtWidgets.QMenu()
        show_action = menu.addAction("Show TorchMoji")
        settings_action = menu.addAction("Settings…")
        menu.addSeparator()
        quit_action = menu.addAction("Quit")

        show_action.triggered.connect(self._main_window.show_main)
        settings_action.triggered.connect(self._open_settings_dialog)
        quit_action.triggered.connect(self.quit)

        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        return tray

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason in {QtWidgets.QSystemTrayIcon.Trigger, QtWidgets.QSystemTrayIcon.DoubleClick}:
            self._main_window.show_main()

    def _maybe_notify_tray(self) -> None:
        if not self._tray_notified:
            self._tray_icon.showMessage(
                "TorchMoji",
                "The window is still running in the system tray.",
                QtWidgets.QSystemTrayIcon.Information,
                3000,
            )
            self._tray_notified = True

    def _on_text_changed(self, text: str) -> None:
        self._current_text = text
        self._update_cli_preview()
        self._prediction_timer.start()

    def _update_cli_preview(self) -> None:
        tokens = build_cli_command(self._settings, self._current_text.strip() or None)
        self._main_window.set_cli_command_tokens(tokens)

    def _run_prediction(self) -> None:
        text = self._current_text.strip()
        if not text:
            self._main_window.clear_results()
            return
        worker = PredictionWorker(text, self._settings)
        self._active_workers.add(worker)
        worker.signals.finished.connect(self._handle_prediction_result)
        worker.signals.failed.connect(self._handle_prediction_error)
        worker.signals.completed.connect(lambda _: self._active_workers.discard(worker))
        self._thread_pool.start(worker)

    @QtCore.Slot(str, object)
    def _handle_prediction_result(self, text: str, result: EmojiPredictionResult) -> None:
        if text != self._current_text.strip():
            return
        self._main_window.show_predictions(result, self._settings.scores)
        self._show_tray_prediction(result)

    @QtCore.Slot(str)
    def _handle_prediction_error(self, message: str) -> None:
        self._main_window.show_error(message)

    def _show_tray_prediction(self, result: EmojiPredictionResult) -> None:
        if not result.selections:
            return
        top = result.selections[0]
        alias = TorchMojiMainWindow._alias_for_selection(top)
        emoji_char = TorchMojiMainWindow._emojize(alias)
        body = f"Top emoji: {emoji_char} ({alias})"
        if self._settings.scores and top.score is not None:
            body += f"\nScore: {top.score:.4f}"
        self._tray_icon.showMessage("TorchMoji Prediction", body, QtWidgets.QSystemTrayIcon.Information, 4000)

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self._settings, self._main_window)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            old_api_settings = (self._settings.api_enabled, self._settings.api_host, self._settings.api_port)
            self._settings = dialog.get_settings()
            save_settings(self._settings)
            self._update_cli_preview()
            self._run_prediction()
            
            # Restart API server if settings changed
            new_api_settings = (self._settings.api_enabled, self._settings.api_host, self._settings.api_port)
            if old_api_settings != new_api_settings:
                self._stop_api_server()
                if self._settings.api_enabled:
                    self._start_api_server()
                else:
                    self._main_window.update_api_status(False)

    def _start_api_server(self) -> None:
        """Start the API server in a background thread."""
        if self._api_server is not None or self._api_thread is not None:
            return
        
        try:
            # Import here to avoid circular dependency
            from ..api_server import TorchMojiAPIServer
            
            self._api_server = TorchMojiAPIServer(
                host=self._settings.api_host,
                port=self._settings.api_port
            )
            
            # Run server in a daemon thread
            self._api_thread = threading.Thread(
                target=self._api_server.run,
                kwargs={"debug": False},
                daemon=True
            )
            self._api_thread.start()
            
            # Update UI
            self._main_window.update_api_status(
                True,
                self._settings.api_host,
                self._settings.api_port
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._main_window.show_error(f"Failed to start API server: {exc}")
            self._api_server = None
            self._api_thread = None

    def _stop_api_server(self) -> None:
        """Stop the API server."""
        if self._api_server is not None:
            # Flask doesn't provide a clean way to stop from another thread,
            # but since we're using daemon threads, they'll stop when the app exits
            self._api_server = None
            self._api_thread = None
            self._main_window.update_api_status(False)

    def run(self) -> int:
        """Start the Qt event loop and return its exit code."""

        return self._qt_app.exec()

    def quit(self) -> None:
        self._stop_api_server()
        self._tray_icon.hide()
        self._main_window.prepare_to_close()
        self._main_window.close()
        self._qt_app.quit()
