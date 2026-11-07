"""Forward obsolete solver executable paths to the current desktop installation."""
from pathlib import Path
import subprocess
import sys


def main():
    executable=Path(sys.executable).resolve()
    target=executable.with_name('AutoAnswer_Win_x64.exe')
    if not target.is_file() or target==executable:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None,
            'The current solver could not be found. Open Sparx Solver (Latest) or reinstall the application.',
            'Sparx Solver launcher',0x10)
        return
    subprocess.Popen([str(target),*sys.argv[1:]],cwd=str(target.parent))


if __name__=='__main__':
    main()
