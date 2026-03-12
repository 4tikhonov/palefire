import sys
import os

def write_csv(filepath, content):
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully wrote CSV to {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 write_csv.py <filepath> <content>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    content = sys.argv[2]
    write_csv(filepath, content)
