import pty
import os
import time
import shutil
import subprocess

pid, fd = pty.fork()
if pid == 0:
    env = os.environ.copy()
    env['TERM'] = 'xterm-256color'
    env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'
    
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
            
    os.execve(gemini_path, ['gemini', '--yolo'], env)
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

