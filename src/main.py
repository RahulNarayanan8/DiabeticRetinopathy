import subprocess
import sys
import time
import os
from datetime import datetime

SCRIPTS = [
    #"cnn.py",
    "vit.py",
    "dino.py",
    "rfc.py",
]


def run_script(script_path: str):
    script_name = os.path.basename(script_path)
    base_name = os.path.splitext(script_name)[0]

    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{base_name}.log")

    print("\n==================================================")
    print(f"Starting {script_name}")
    print(f"Logging to: {log_path}")
    print("==================================================\n")

    start_time = time.time()

    with open(log_path, "w") as log_file:
        log_file.write(f"=== {script_name} started at {timestamp} ===\n\n")
        log_file.flush()

        # Start the subprocess and capture its stdout+stderr
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream line-by-line to both console and log file
        for line in process.stdout:
            # print to console (prefix with script name so you know the source)
            sys.stdout.write(f"[{base_name}] {line}")
            sys.stdout.flush()

            # write to log file
            log_file.write(line)
            log_file.flush()

        process.stdout.close()
        return_code = process.wait()

    end_time = time.time()
    duration_min = (end_time - start_time) / 60.0

    if return_code != 0:
        print(f"\n{script_name} FAILED with exit code {return_code}")
        print(f"Check log: {log_path}\n")
        sys.exit(return_code)

    print(f"\n{script_name} completed successfully in {duration_min:.2f} minutes.")
    print(f"Log saved to: {log_path}\n")


def main():
    print("\n==================================================")
    print(" Running CNN -> ViT -> DINO -> RFC sequentially")
    print(" Single-GPU safe run (one script at a time)")
    print(" Logs will be written to ./logs/")
    print("==================================================\n")

    for script in SCRIPTS:
        run_script(script)

    print("\n==================================================")
    print("All models finished successfully.")
    print("Check ./logs/ for full outputs from each run.")
    print("==================================================\n")


if __name__ == "__main__":
    main()
