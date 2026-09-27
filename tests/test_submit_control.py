"""The "submit" control command outside long-form: one key starts, and stops with Enter."""

import unittest
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated


class FakeConfig:
    def __init__(self, mode):
        self.mode = mode

    def get_setting(self, key, default=None):
        return self.mode if key == 'recording_mode' else default


class SubmitControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, is_recording, mode='toggle'):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig(mode)
        app.is_recording = is_recording
        app._start_recording = mock.Mock()
        app._stop_recording = mock.Mock()
        app._autostop_start_silence_monitor = mock.Mock()
        app._longform = mock.Mock()
        return app

    def test_submit_while_recording_stops_and_presses_enter(self):
        app = self._app(is_recording=True)
        app._handle_control_command('submit')
        app._stop_recording.assert_called_once_with(submit=True)
        app._start_recording.assert_not_called()

    def test_submit_while_idle_starts_recording(self):
        app = self._app(is_recording=False)
        app._handle_control_command('submit')
        app._start_recording.assert_called_once()
        app._stop_recording.assert_not_called()

    def test_long_form_submit_is_unchanged(self):
        app = self._app(is_recording=True, mode='long_form')
        app._handle_control_command('submit')
        app._longform.submit_shortcut.assert_called_once_with()
        app._stop_recording.assert_not_called()


if __name__ == '__main__':
    unittest.main()
