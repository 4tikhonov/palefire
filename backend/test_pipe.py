import subprocess
import os
import json
import shutil

env = os.environ.copy()
env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin:/Users/vyacheslavtykhonov/.nvm/versions/node/v20.12.2/bin'

gemini_path = shutil.which('gemini', path=env['PATH'])
if not gemini_path:
    for p in ['/opt/homebrew/bin/gemini', '/usr/local/bin/gemini', '/usr/bin/gemini', os.path.expanduser('~/.local/bin/gemini')]:
        if os.path.exists(p):
            gemini_path = p
            break
    else:
        gemini_path = 'gemini'

process = subprocess.Popen(
    [gemini_path, '--yolo', '-o', 'stream-json'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=env,
    text=True
)

process.stdin.write("What is 1+1?\n")
process.stdin.flush()

for i in range(10):
    line = process.stdout.readline()
    print(f"[{i}]: {repr(line)}")
    if not line:
        break

process.terminate()
