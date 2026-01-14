#!/usr/bin/env python3
# INDEFINITE recording of 10 minute clips till MIN_FREE_GB or Ctrl-C
import os, shutil, subprocess
from datetime import datetime
VIDEO_DIR = "/media/user/disk/videos"
FALLBACK_DIR = "/home/user/videos"
LOG_FILE = "/var/log/rpicam/rpicam.log"
# cam settings
SEG_MS = "600000" # 8 Hours
FPS = "15"
BITRATE = "2000000" # 2 Mbps
# disk guard
MIN_FREE_GB = 100 # 5% of 2TB
def main():
    rpiVIDSec()
def rpiVIDSec():
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    out_dir = VIDEO_DIR if os.path.ismount("/media/user/disk") else FALLBACK_DIR
    out_file = f"{out_dir}/videos_{session_ts}.mp4"
    seq = 0
    with open(LOG_FILE, "ab", buffering=0) as log:
        log.write(f"\n=== Vid-Sec Started: {session_ts} ===\n".encode())
        log.write(f"OUT: {out_file}\n".encode())
        try:
            while True:
                free_gb = shutil.disk_usage(out_dir).free / (1024**3)
                if free_gb < MIN_FREE_GB:
                    log.write(f"\n=== STOP: low disk space ({free_gb:.2f} GB free < {MIN_FREE_GB} GB)===\n".encode())
                    break
                clip_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                mp4 = f"{out_dir}/sec_{session_ts}_{seq:04d}_{clip_ts}.mp4"
                cmd = [
                    "rpicam-vid",
                    "--timeout", SEG_MS
                    "--nopreview",
                    "--codec", "h264",
                    "--framerate", FPS,
                    "--bitrate", BITRATE,
                    "--intra", FPS,
                    "--inline",
                    "-o", mp4,
                ]
                log.write(f"\nCMD: {' '.join(cmd)}\n".encode())
                subprocess.run(cmd, stdout=log,stderr=log, check=True)
                seq += 1
        except KeyboardInterrupt:
            log.write(b"=== Recording interrupted bu user (Ctrl+C) ===\n")
            raise
        finally:
            end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log.write(f"=== Vid-Sec ended (session {session_ts}, ended {end_ts}) ===\n".encode())
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
