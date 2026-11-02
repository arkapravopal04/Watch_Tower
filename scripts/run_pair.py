"""Run one command per GPU (at most two T4s), stream both logs live, keep every exit code.

Usage: python scripts/run_pair.py --out LOGDIR '["python", "-m", "pcl", ...]' ['[...]']
Each command is a JSON argument list and sees exactly one GPU (CUDA_VISIBLE_DEVICES).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def _pump(process, handle, prefix):
    for line in process.stdout:
        handle.write(line)
        handle.flush()
        sys.stdout.write(f'{prefix} {line}')
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('commands', nargs='+')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if len(args.commands) > 2:
        parser.error('at most one command per GPU (two T4s)')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    processes, handles, threads = [], [], []
    try:
        for gpu, command in enumerate(args.commands):
            handle = (output / f'gpu{gpu}.log').open('w', encoding='utf-8')
            handles.append(handle)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'TOKENIZERS_PARALLELISM': 'false',
                   'PYTHONUNBUFFERED': '1'}
            process = subprocess.Popen(json.loads(command), env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            processes.append(process)
            thread = threading.Thread(target=_pump, args=(process, handle, f'[gpu{gpu}]'), daemon=True)
            thread.start()
            threads.append(thread)
        codes = [process.wait() for process in processes]
        for thread in threads:
            thread.join()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for handle in handles:
            handle.close()
    (output / 'exit_codes.json').write_text(json.dumps(dict(enumerate(codes))))
    print('exit codes:', codes)
    raise SystemExit(0 if all(code == 0 for code in codes) else 1)


if __name__ == '__main__':
    main()
