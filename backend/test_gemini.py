import pty
import os
import time

pid, fd = pty.fork()
if pid == 0:
    env = os.environ.copy()
    env['TERM'] = 'xterm-256color'
    env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'
    os.execve('/opt/homebrew/bin/gemini', ['gemini', '--yolo'], env)
else:
    import fcntl, termios, struct
    winsize = struct.pack("HHHH", 24, 80, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
    for _ in range(3):
        time.sleep(1)
        try:
            print("Read:", os.read(fd, 1024))
        except OSError as e:
            print("OSError", e)
            break
    print("Waitstatus:", os.waitpid(pid, os.WNOHANG))

