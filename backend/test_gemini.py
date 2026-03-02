import pty
import os
import time
import shutil
import subprocess
import sys

pid, fd = pty.fork()
if pid == 0:
    env = os.environ.copy()
    env['TERM'] = 'xterm-256color'
    
    # Try to dynamically extend PATH
    node_paths = []
    if sys.platform == 'darwin':
        node_paths.append('/opt/homebrew/bin')
    node_paths.extend([
        '/usr/local/bin', os.path.expanduser('~/.npm-global/bin')
    ])
    try:
        nvm_dir = os.path.expanduser('~/.nvm/versions/node')
        if os.path.isdir(nvm_dir):
            versions = sorted(os.listdir(nvm_dir), reverse=True)
            if versions:
                node_paths.append(os.path.join(nvm_dir, versions[0], 'bin'))
    except Exception:
        pass
        
    for np in node_paths:
        if os.path.isdir(np) and np not in env.get('PATH', '').split(':'):
            env['PATH'] = env.get('PATH', '') + f':{np}'
    
    gemini_path = None
    if sys.platform == 'darwin':
        search_paths = [
            '/opt/homebrew/bin/gemini', '/usr/local/bin/gemini', 
            os.path.expanduser('~/.npm-global/bin/gemini'),
            '/usr/bin/gemini', os.path.expanduser('~/.local/bin/gemini')
        ]
    else:
        search_paths = [
            '/usr/local/bin/gemini', os.path.expanduser('~/.npm-global/bin/gemini'),
            '/usr/bin/gemini', '/bin/gemini', 
            os.path.expanduser('~/.local/bin/gemini'), os.path.expanduser('~/bin/gemini'), '/opt/gemini/bin/gemini'
        ]
    for p in search_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            gemini_path = p
            break
            
    if not gemini_path:
        gemini_path = shutil.which('gemini', path=env['PATH'])

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

