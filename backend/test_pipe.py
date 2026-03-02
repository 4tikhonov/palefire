import subprocess
import os
import json
import shutil

env = os.environ.copy()
env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin:/Users/vyacheslavtykhonov/.nvm/versions/node/v20.12.2/bin'

gemini_path = shutil.which('gemini', path=env['PATH'])
if not gemini_path:
    search_paths = [
        '/opt/homebrew/bin/gemini', '/usr/local/bin/gemini', '/usr/bin/gemini', '/bin/gemini', 
        os.path.expanduser('~/.local/bin/gemini'), os.path.expanduser('~/bin/gemini'), '/opt/gemini/bin/gemini'
    ]
    for p in search_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            gemini_path = p
            break
    else:
        try:
            which_result = subprocess.run(['which', 'gemini'], capture_output=True, text=True)
            if which_result.returncode == 0:
                gemini_path = which_result.stdout.strip()
        except Exception:
            pass
            
if not gemini_path:
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
