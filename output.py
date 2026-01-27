import subprocess
import os

# Third attempt: Highly stable Veritasium video URL: "The Simplest Math Problem No One Can Solve"
video_url = "https://www.youtube.com/watch?v=094y1Z2wpJg" 
applescript = f'open location "{video_url}"'

# Ensure correct path handling for subprocess
expanded_cwd = os.path.expanduser('~')

print(f"Third attempt: Opening a different Veritasium video: {video_url}")

# Execute the AppleScript command to open the video in the default browser.
subprocess.run(['osascript', '-e', applescript], cwd=expanded_cwd)