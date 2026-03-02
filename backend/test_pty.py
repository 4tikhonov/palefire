import pty
import os
import shutil

pid, fd = pty.fork()
if pid == 0:
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

    os.execve(gemini_path, ['gemini', '--yolo', '-o', 'stream-json'], env)
else:
    os.write(fd, b"What is 2+2?\n")
    import time
    time.sleep(3)
    out = os.read(fd, 8192)
    print(repr(out))
