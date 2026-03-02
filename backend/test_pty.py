import pty
import os
import shutil
import subprocess

pid, fd = pty.fork()
if pid == 0:
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

    os.execve(gemini_path, ['gemini', '--yolo', '-o', 'stream-json'], env)
else:
    os.write(fd, b"What is 2+2?\n")
    import time
    time.sleep(3)
    out = os.read(fd, 8192)
    print(repr(out))
