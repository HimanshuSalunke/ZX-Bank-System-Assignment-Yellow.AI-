"""
ZX Bank AI — One-Command Startup Script.

Usage:
    python run.py

This script:
  1. Sets HuggingFace offline mode (no network requests for cached models)
  2. Kills any existing process on port 8000
  3. Checks if indexes exist (runs setup_index.py if not)
  4. Starts the FastAPI server with uvicorn
"""

import os
import subprocess
import sys
import socket
import time

# ── CRITICAL: Set offline mode BEFORE any HuggingFace imports ──────────
# This prevents sentence-transformers from trying to access huggingface.co
# on every startup. Models are loaded from local cache only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def kill_port_8000():
    """Kill any process using port 8000."""
    print("[*] Checking for existing processes on port 8000...")
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        killed = False
        for line in result.stdout.splitlines():
            if ":8000" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                print(f"[*] Killing PID {pid} on port 8000...")
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, timeout=5,
                    )
                    print(f"[✓] Killed PID {pid}")
                    killed = True
                except Exception:
                    pass
        if killed:
            time.sleep(1)  # Wait for port release
        else:
            print("[✓] Port 8000 is free")
    except Exception:
        pass


def is_port_free(port=8000):
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def check_indexes():
    """Check if FAISS and BM25 indexes exist, build if missing."""
    indexes_dir = os.path.join(PROJECT_ROOT, "indexes")
    faiss_path = os.path.join(indexes_dir, "faiss_index.bin")
    bm25_path = os.path.join(indexes_dir, "bm25_index.pkl")

    if os.path.exists(faiss_path) and os.path.exists(bm25_path):
        print("[✓] Indexes found — skipping rebuild")
        return True

    print("[*] Indexes not found — building now...")
    print("    (This only happens once. It processes 20 documents into")
    print("    FAISS + BM25 indexes using GPU-accelerated embeddings.)")
    # Index build needs online mode for first-time model download
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "setup_index.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )
    if result.returncode != 0:
        print("[✗] Index build failed!")
        return False
    print("[✓] Indexes built successfully")
    return True


def main():
    """Main entry point — kill old processes, check indexes, start server."""
    print("=" * 60)
    print("  ZX Bank AI — Starting System")
    print("=" * 60)
    print(f"  Model: {os.environ.get('LLM_MODEL', 'openai/gpt-4.1-nano')}")
    print(f"  HF Offline: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")
    print("=" * 60)

    # Step 1: Kill existing processes on port 8000
    kill_port_8000()

    if not is_port_free():
        print("[!] Port 8000 still in use — waiting 3s...")
        time.sleep(3)
        if not is_port_free():
            print("[✗] Cannot start: port 8000 is still occupied")
            sys.exit(1)

    # Step 2: Check/build indexes
    if not check_indexes():
        sys.exit(1)

    # Step 3: Start the server
    print("\n" + "=" * 60)
    print("  Server starting on http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "app.main:app",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--log-level", "info",
            ],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),  # Pass the offline mode env vars
        )
    except KeyboardInterrupt:
        print("\n[*] Server stopped by user")


if __name__ == "__main__":
    main()
