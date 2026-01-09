#!/usr/bin/env python3
# Take a 5 second test video with rpicam
import os, subprocess
from datetime import datetime
VIDEO_DIR = "/media/user/disk/videos"
FALLBACK_DIR = "/home/user/videos"
LOG_FILE = "/var/log/rpicam/rpicam.log"
def rpiVIDTest():
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    out_dir = VIDEO_DIR if os.path.ismount("/media/user/disk") else FALLBACK_DIR
    out_file = f"{out_dir}/videos_{session_ts}.mp4"
    cmd = [
        "rpicam-vid",
        "--timeout", "6000", #5 second video
        "--nopreview",
        "--codec", "h264",
        "-o", out_file,
    ]
    with open(LOG_FILE, "ab", buffering=0) as log:
        log.write(f"\n=== Vid-Test Started: {session_ts} ===\n".encode())
        log.write(f"OUT: {out_file}\n".encode())
        try:
            subprocess.run(cmd, stdout=log,stderr=log, check=True)
        except KeyboardInterrupt:
            log.write(b"=== Recording interrupted bu user (Ctrl+C) ===\n")
            raise
        finally:
            end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log.write(f"=== Vid-Test (session {session_ts}, ended {end_ts}) ===\n".encode())
    print(out_file)
def main():
    rpiVIDTest()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrup:
        exit()
