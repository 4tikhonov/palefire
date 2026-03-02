import pty
import os

pid, fd = pty.fork()
if pid == 0:
    env = os.environ.copy()
    env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin:/Users/vyacheslavtykhonov/.nvm/versions/node/v20.12.2/bin'
    os.execve('/opt/homebrew/bin/gemini', ['gemini', '--yolo', '-o', 'stream-json'], env)
else:
    os.write(fd, b"What is 2+2?\n")
    import time
    time.sleep(3)
    out = os.read(fd, 8192)
    print(repr(out))
