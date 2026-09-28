"""The Waybar module's `follow` loop reacts to the service's state files at once."""
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAY = ROOT / 'config/hyprland/hyprwhspr-tray.sh'
TICK = 2  # follow's full-check timeout; a reaction faster than this came from inotify


@unittest.skipUnless(shutil.which('inotifywait'), 'needs inotify-tools')
class TrayFollowTests(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        stubs = tmp / 'bin'
        stubs.mkdir()
        # The service is "active"; everything else about the host is absent
        for name in ('systemctl', 'pactl'):
            (stubs / name).write_text('#!/bin/sh\nexit 0\n')
            (stubs / name).chmod(0o755)
        self.runtime = tmp / 'run' / 'hyprwhspr'
        env = dict(os.environ, PATH=f"{stubs}:{os.environ['PATH']}", HOME=str(tmp),
                   XDG_RUNTIME_DIR=str(tmp / 'run'), XDG_CONFIG_HOME=str(tmp / 'config'))
        self.proc = subprocess.Popen(['bash', str(TRAY), 'follow'], stdout=subprocess.PIPE,
                                     text=True, env=env, start_new_session=True)
        self.addCleanup(self.proc.wait)
        self.addCleanup(os.killpg, self.proc.pid, signal.SIGTERM)
        self.lines = queue.Queue()
        threading.Thread(target=lambda: [self.lines.put(line) for line in self.proc.stdout],
                         daemon=True).start()

    def wait_for(self, predicate, within):
        deadline = time.monotonic() + within
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                status = json.loads(self.lines.get(timeout=remaining))
            except queue.Empty:
                break
            if predicate(status['class']):
                return
        self.fail(f'no matching status line within {within}s')

    def test_state_file_changes_show_before_the_next_tick(self):
        # Both states come from service-written files alone, so the stubbed-out
        # mic checks (and their error debounce) never decide the outcome
        self.wait_for(lambda _: True, within=10)  # initial state, whatever it is
        unloaded = self.runtime / 'model_unloaded'
        unloaded.touch()
        self.wait_for(lambda cls: cls == 'unloaded', within=TICK * 0.75)
        unloaded.unlink()
        (self.runtime / 'recording_status').write_text('true')
        self.wait_for(lambda cls: cls == 'recording', within=TICK * 0.75)


if __name__ == '__main__':
    unittest.main()
