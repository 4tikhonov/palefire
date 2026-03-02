import subprocess
import os
import json

env = os.environ.copy()
env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin:/Users/vyacheslavtykhonov/.nvm/versions/node/v20.12.2/bin'

process = subprocess.Popen(
    ['/opt/homebrew/bin/gemini', '--yolo', '-o', 'stream-json'],
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
