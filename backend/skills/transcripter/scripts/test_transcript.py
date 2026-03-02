import subprocess
import os
import sys

def test_transcript():
    script_path = os.path.join(os.path.dirname(__file__), 'get_transcript.py')
    video_id = "b6ucRt-rChI" # ID from https://www.youtube.com/watch?v=b6ucRt-rChI
    
    print(f"Testing transcript extraction for video ID: {video_id}")
    try:
        result = subprocess.run(
            [sys.executable, script_path, video_id],
            capture_output=True,
            text=True,
            check=True
        )
        print("Transcript extraction successful!")
        print("Output snippet:")
        # Print the first 500 characters of the transcript to verify
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        if len(result.stdout.strip()) == 0:
            print("Error: The extracted transcript is empty.")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"Transcript extraction failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    test_transcript()
