"""Local diagnostics that survive a stopped run or an unexpected process failure."""
import datetime
import faulthandler
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time

_fault_streams = []


class InterfaceWatchdog:
    """Record all thread stacks if Tk stops responding; never touch Tk from a worker."""
    def __init__(self, logger, path, timeout=20):
        self.logger = logger
        self.path = Path(path).with_suffix('.hang.log')
        self.timeout = timeout
        self.last_beat = time.monotonic()
        self.reported = False
        self.stopped = threading.Event()

    def beat(self):
        self.last_beat = time.monotonic()
        self.reported = False

    def check(self):
        if not self.reported and time.monotonic() - self.last_beat > self.timeout:
            self.reported = True
            with self.path.open('a', encoding='utf-8') as stream:
                faulthandler.dump_traceback(file=stream, all_threads=True)
            self.logger.error('Interface unresponsive for over %ss; thread stacks: %s', self.timeout, self.path)

    def start(self):
        def watch():
            while not self.stopped.wait(5):
                try:
                    self.check()
                except OSError:
                    self.logger.exception('Could not write interface hang diagnostics')
        threading.Thread(target=watch, name='interface-watchdog', daemon=True).start()

    def close(self):
        self.stopped.set()


class _PrivateFormatter(logging.Formatter):
    def format(self, record):
        return re.sub(r'\bsk-[A-Za-z0-9_-]+', '[API KEY REDACTED]', super().format(record))


def start_session_log(directory=None, native_traces=True):
    """Keep ordinary events and Python/native failure traces outside the repository."""
    root = Path(directory or Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.local/share'))) / 'AutoAnswer' / 'logs')
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = root / f'solver_{stamp}_{os.getpid()}.log'
    logger = logging.getLogger('autoanswer.' + str(path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(_PrivateFormatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    if native_traces:
        try:
            stream = path.with_suffix('.fault.log').open('a', encoding='utf-8')
            faulthandler.enable(file=stream, all_threads=True)
            _fault_streams.append(stream)
        except (OSError, RuntimeError):
            logger.exception('Could not enable native failure diagnostics')
    original = sys.excepthook
    def exception(kind, value, trace):
        logger.critical('Unhandled application exception', exc_info=(kind,value,trace))
        original(kind,value,trace)
    sys.excepthook = exception
    def worker_exception(args):
        logger.error('Unhandled worker exception: %s', args.thread.name,
                     exc_info=(args.exc_type,args.exc_value,args.exc_traceback))
    threading.excepthook = worker_exception
    logger.info('Solver process started: pid=%s', os.getpid())
    return logger, path
