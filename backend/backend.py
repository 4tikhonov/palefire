import asyncio
import websockets
import subprocess
import json
import os
import re
import shutil
import sys

# Paths dynamically resolved relative to this backend.py script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Try to find 'footnotes' as a sibling (standard for the extension dev environment)
FOOTNOTES_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '../footnotes'))
BASE_DIR = FOOTNOTES_DIR if os.path.isdir(FOOTNOTES_DIR) else PROJECT_ROOT
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'history.json')

# Global State to persist across client reconnects
global_state = {
    'task': None,
    'prompt': None,
    'result_type': None,
    'result_data': None,
    'is_delivered': True,
    'clients': set(),
    'pages': []
}

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def get_gemini_path(env=None):
    # Advanced search locations based on OS prioritized FIRST
    if sys.platform == 'darwin':
        search_paths = [
            '/opt/homebrew/bin/gemini',     # MacOS Apple Silicon Homebrew
            '/usr/local/bin/gemini',        # MacOS Intel & common Node
            os.path.expanduser('~/.npm-global/bin/gemini'),
            '/usr/bin/gemini',              
            os.path.expanduser('~/.local/bin/gemini')
        ]
    else:
        search_paths = [
            '/usr/local/bin/gemini',        # Linux primary Node bin
            os.path.expanduser('~/.npm-global/bin/gemini'),
            '/usr/bin/gemini',              
            '/bin/gemini',                  
            os.path.expanduser('~/.local/bin/gemini'), 
            os.path.expanduser('~/bin/gemini'),        
            '/opt/gemini/bin/gemini'        
        ]
    
    for p in search_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    if env and 'PATH' in env:
        path = shutil.which('gemini', path=env['PATH'])
        if path:
            return path
            
    # Attempt to locate via system 'which' fallback
    try:
        which_result = subprocess.run(['which', 'gemini'], capture_output=True, text=True)
        if which_result.returncode == 0:
            return which_result.stdout.strip()
    except Exception:
        pass
        
    return 'gemini'

global_state['pages'] = load_cache()

async def broadcast(message_dict):
    if not global_state['clients']:
        return
    message_str = json.dumps(message_dict)
    # create a copy of clients to safely iterate
    for ws in list(global_state['clients']):
        try:
            await ws.send(message_str)
        except websockets.exceptions.ConnectionClosed:
            global_state['clients'].remove(ws)

def run_gemini(prompt_input, env):
    skills_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.agent/skills'))
    gemini_path = get_gemini_path(env)
    
    # 1. Attempt with session restoration
    cmd_resume = [gemini_path, '-p', prompt_input, '--yolo', '-o', 'json', '-r', 'latest', '--policy', skills_dir]
    try:
        print(f"[DEBUG] CLI Call (Attempt 1): {' '.join(cmd_resume)}")
        result = subprocess.run(cmd_resume, env=env, capture_output=True, text=True, cwd=BASE_DIR)
        
        # Check for ANY signs of session resume failure
        combined_output = (result.stderr + result.stdout).lower()
        needs_fallback = any(msg in combined_output for msg in [
            "no sessions found", "no previous sessions found", 
            "error resuming session", "failed to resume", 
            "no session"
        ])
        
        # If the resume call failed OR didn't produce JSON
        if needs_fallback or (result.returncode != 0 and "response" not in result.stdout) or not re.search(r'\{.*\}', result.stdout, re.DOTALL):
            print("[DEBUG] Session restoration failed or invalid output, falling back to fresh session...")
            # 2. Fresh session (no session restoration)
            cmd_fresh = [gemini_path, '-p', prompt_input, '--yolo', '-o', 'json', '--policy', skills_dir]
            print(f"[DEBUG] CLI Call (Attempt 2 - Fresh): {' '.join(cmd_fresh)}")
            result = subprocess.run(cmd_fresh, env=env, capture_output=True, text=True, cwd=BASE_DIR)
            
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", f"Gemini CLI tool '{gemini_path}' was not found. Please ensure it is installed and in your PATH.", 1
    except Exception as e:
        return "", f"Execution error: {str(e)}", 1

async def background_gemini_task(prompt_text, env):
    loop = asyncio.get_event_loop()
    
    # Check if prompt contains youtube URL
    if "youtube.com/watch" in prompt_text or "youtu.be/" in prompt_text:
        match = re.search(r'(?:v=|youtu\.be/)([\w-]+)', prompt_text)
        if match:
            video_id = match.group(1)
            transcript_cache_file = os.path.join(CACHE_DIR, f'transcript_{video_id}.txt')
            transcript_text = None
            
            if os.path.exists(transcript_cache_file):
                try:
                    with open(transcript_cache_file, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
                    print(f"Loaded transcript for {video_id} from cache.")
                except Exception as e:
                    print(f"Failed to read transcript cache: {e}")

            if not transcript_text:
                try:
                    from youtube_transcript_api import YouTubeTranscriptApi
                    t_list = YouTubeTranscriptApi().list(video_id)
                    try:
                        transcript_obj = t_list.find_transcript(['en'])
                    except Exception:
                        # Fallback to the first available transcript
                        for t in t_list:
                            transcript_obj = t
                            break
                    fetched = transcript_obj.fetch()
                    # Text attribute requires dict access in older version but objects in newer versions. 
                    # Let's handle both gracefully:
                    transcript_text = " ".join([t['text'] if isinstance(t, dict) else t.text for t in fetched])
                    transcript_text = transcript_text[:30000] # Limit to ~10k words
                    
                    # Save to cache
                    if not os.path.exists(CACHE_DIR):
                        os.makedirs(CACHE_DIR)
                    with open(transcript_cache_file, 'w', encoding='utf-8') as f:
                        f.write(transcript_text)
                    print(f"Saved transcript for {video_id} to cache.")
                except Exception as e:
                    prompt_text = f"{prompt_text}\n\n[FAILED TO INJECT YOUTUBE TRANSCRIPT]: {str(e)}"
            
            if transcript_text:
                prompt_text = f"{prompt_text}\n\nYou MUST use this extracted video transcript as the canonical content of the video:\n[YOUTUBE TRANSCRIPT]:\n{transcript_text}"
    
    # Execute blocking operation in an executor so the event loop remains unblocked
    stdout_data, stderr_data, returncode = await loop.run_in_executor(None, run_gemini, prompt_text, env)
    
    # Process Results
    # Use re.DOTALL and search for the LAST json-like block to avoid banner noise
    json_blocks = re.findall(r'(\{.*?\})', stdout_data, re.DOTALL)
    if json_blocks:
        # Take the last block which is most likely the actual response
        json_str = json_blocks[-1]
        try:
            parsed = json.loads(json_str)
            response_text = parsed.get("response", "No response parsed.")
            
            global_state['result_type'] = 'response'
            global_state['result_data'] = response_text
        except json.JSONDecodeError:
            global_state['result_type'] = 'error'
            global_state['result_data'] = f"Failed to parse JSON response: {json_str[:200]}..."
    else:
        err_msg = stderr_data if stderr_data else stdout_data
        if "response" not in stdout_data and len(stdout_data.strip()) > 0:
            global_state['result_type'] = 'response'
            global_state['result_data'] = stdout_data.strip()
        else:
            global_state['result_type'] = 'error'
            global_state['result_data'] = f"Error from Gemini CLI: {err_msg}"

    global_state['is_delivered'] = False
    
    # Broadcast to all connected clients immediately!
    await broadcast({'type': global_state['result_type'], 'data': global_state['result_data']})
    global_state['is_delivered'] = True
    global_state['task'] = None

async def chat_handler(websocket):
    global_state['clients'].add(websocket)
    print("New chat client connected. Active clients:", len(global_state['clients']))

    # 1. On connect, send history cache immediately!
    try:
        await websocket.send(json.dumps({'type': 'cache_data', 'data': global_state['pages']}))
    except:
        pass

    # 2. On connect, restore state! (crucial for Chrome/Opera extensions dropping background websockets)
    if global_state['task'] is not None and not global_state['task'].done():
        print(f"Restoring running task for prompt: {global_state['prompt'][:50]}")
        try:
            await websocket.send(json.dumps({'type': 'restore_running', 'data': global_state['prompt']}))
        except:
            pass
    elif not global_state['is_delivered'] and global_state['result_data'] is not None:
        print("Delivering missed response to new client.")
        try:
            await websocket.send(json.dumps({'type': global_state['result_type'], 'data': global_state['result_data']}))
            global_state['is_delivered'] = True
        except:
            pass

    env = os.environ.copy()
    
    node_paths = []
    if sys.platform == 'darwin':
        node_paths.append('/opt/homebrew/bin')
    node_paths.extend([
        '/usr/local/bin',
        os.path.expanduser('~/.npm-global/bin'),
        os.path.expanduser('~/.nvm/versions/node/current/bin')
    ])
    
    # Try to add actual nvm node path if present
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

    gemini_path = get_gemini_path(env)
    if not shutil.which(gemini_path, path=env['PATH']):
        try:
            warning_msg = f"⚠️ System Check Failed: 'gemini' CLI tool not found on the backend path: {gemini_path}. The extension will not function correctly until you install the AI Footnotes CLI and ensure it is accessible in your environment."
            await websocket.send(json.dumps({'type': 'error', 'data': warning_msg}))
        except:
            pass

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                if msg.get('type') == 'input' and 'data' in msg:
                    prompt_text = msg['data']
                    print(f"Received prompt: {prompt_text[:50]}...")

                    if global_state['task'] and not global_state['task'].done():
                        await websocket.send(json.dumps({'type': 'error', 'data': 'A previous task is already running. Please wait for it to finish...'}))
                        continue

                    global_state['prompt'] = prompt_text
                    global_state['is_delivered'] = False

                    # Start the background execution task globally
                    global_state['task'] = asyncio.create_task(background_gemini_task(prompt_text, env))

                elif msg.get('type') == 'keepalive':
                    pass
                elif msg.get('type') == 'save_cache':
                    global_state['pages'] = msg.get('data', [])
                    if not os.path.exists(CACHE_DIR):
                        os.makedirs(CACHE_DIR)
                    with open(CACHE_FILE, 'w') as f:
                        json.dump(global_state['pages'], f)
            except json.JSONDecodeError:
                pass
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"Error handling message: {e}")
                try:
                    await websocket.send(json.dumps({'type': 'error', 'data': f"System error: {str(e)}"}))
                except:
                    pass

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"Error in connection loop: {e}")
    finally:
        print("Chat client disconnected.")
        if websocket in global_state['clients']:
            global_state['clients'].remove(websocket)

async def keepalive_loop():
    while True:
        await asyncio.sleep(2.0)
        await broadcast({'type': 'keepalive'})

async def main():
    print(f"Starting Pale Fire Footnotes JSON RPC Backend on ws://127.0.0.1:8775 (CWD: {BASE_DIR})")
    
    # Startup check for gemini CLI with dynamic path expansion
    env = os.environ.copy()
    
    node_bin_paths = []
    if sys.platform == 'darwin':
        node_bin_paths.append('/opt/homebrew/bin')
    node_bin_paths.extend(['/usr/local/bin', os.path.expanduser('~/.npm-global/bin')])
    
    try:
        nvm_dir = os.path.expanduser('~/.nvm/versions/node')
        if os.path.isdir(nvm_dir):
            versions = sorted(os.listdir(nvm_dir), reverse=True)
            if versions:
                node_bin_paths.append(os.path.join(nvm_dir, versions[0], 'bin'))
    except:
        pass

    path_addition = ':'.join(p for p in node_bin_paths if os.path.isdir(p))
    if path_addition:
        env['PATH'] = env.get('PATH', '') + f':{path_addition}'
        
    gemini_path = get_gemini_path(env)
    if not shutil.which(gemini_path, path=env['PATH']) and not os.path.exists(gemini_path):
        print(f"\n[WARNING] 'gemini' CLI not found. Resolved path attempted: {gemini_path}")
        print("Please ensure you have installed the gemini CLI and it is accessible in your environment.")
        print("The extension will not function correctly until the CLI is available.\n")
    else:
        print(f"Gemini CLI found at: {shutil.which(gemini_path, path=env['PATH']) or gemini_path}")
        
    # Start the keepalive loop in the background
    asyncio.create_task(keepalive_loop())
    
    # Run the server with native pings disabled from the library so we rely solely on our broadcasted JSON keepalives
    async with websockets.serve(chat_handler, "127.0.0.1", 8775, ping_interval=None, ping_timeout=None):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
