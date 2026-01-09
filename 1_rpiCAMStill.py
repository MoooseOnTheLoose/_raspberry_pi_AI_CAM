#!/usr/bin/env python3
# Take a single picture with rpicam
import os, subprocess
from datetime import datetime
IMAGE_DIR = "/media/user/disk/images"
FALLBACK_DIR = "/home/user/images"
LOG_FILE = "/var/log/rpicam/rpicam.log"
# cam settings
SEG_MS = "2000" #2 seconds for settle
def main():
    rpiCAMStill()
def rpiCAMStill():
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    out_dir = IMAGE_DIR if os.path.ismount("/media/user/disk") else FALLBACK_DIR
    output_file = f"{out_dir}/image_{session_ts}.jpg"
    with open(LOG_FILE, "ab", buffering=0) as log:
        log.write(f"\n=== Still Capture Started: {session_ts} ===\n".encode())
        try:
            subprocess.run(
            [
                "rpicam-still",
                "--timeout", SEG_MS,
                "--nopreview",
                "-o", output_file,
            ],
            stdout=log,
            stderr=log,
            check=True,
            )
        except KeyboardInterrupt:
            log.write(b"=== Recording interrupted bu user (Ctrl+C) ===\n")
            raise
        finally:
            end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log.write(f"=== Still Capture ended (session {session_ts}, ended {end_ts}) ===\n".encode())
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrup:
        exit()
