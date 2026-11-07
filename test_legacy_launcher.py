"""Obsolete executable names must launch the current solver, never recurse."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import legacy_launcher


class LegacyLauncherTests(unittest.TestCase):
    def test_old_path_forwards_arguments_to_current_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            target=root/'AutoAnswer_Win_x64.exe';target.touch()
            with patch('legacy_launcher.sys.executable',str(root/'old_backup.exe')), \
                 patch('legacy_launcher.sys.argv',['old_backup.exe','--background-browser']), \
                 patch('legacy_launcher.subprocess.Popen') as launch:
                legacy_launcher.main()
            launch.assert_called_once_with([str(target),'--background-browser'],cwd=str(root))

    def test_missing_target_or_canonical_launcher_never_spawns_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            popup=Mock()
            for name in ('old_backup.exe','AutoAnswer_Win_x64.exe'):
                if name=='AutoAnswer_Win_x64.exe':(root/name).touch()
                with patch('legacy_launcher.sys.executable',str(root/name)), \
                     patch('legacy_launcher.subprocess.Popen') as launch, \
                     patch('ctypes.windll',SimpleNamespace(user32=SimpleNamespace(MessageBoxW=popup)),create=True):
                    legacy_launcher.main()
                    launch.assert_not_called()
            self.assertEqual(popup.call_count,2)


if __name__=='__main__':unittest.main()
