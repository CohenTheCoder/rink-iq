"""Publish this repo as a live demo on Hugging Face Spaces (free CPU hardware).

    pip install -U "huggingface_hub[cli]"
    hf auth login                      # paste a token with "write" access (huggingface.co/settings/tokens)
    python scripts/deploy_space.py     # → https://huggingface.co/spaces/<you>/<SPACE_NAME>

Options: --private (only you can open it) · --name other-name · --dry-run (build the folder, don't upload)

What it does: copies the files git tracks into a temporary folder, adds a Dockerfile and the Space
settings header Hugging Face needs at the top of README.md, then uploads the folder. Re-run it to update.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = {'name': 'rink-iq', 'title': 'RinkIQ · hockey film analytics', 'emoji': '🏒', 'from': 'blue', 'to': 'red', 'description': 'Hockey film → tracking, speeds, shots & report cards'}

DOCKERFILE = """FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg git \\
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \\
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics TRANSFORMERS_NO_TF=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
# CPU-only PyTorch keeps the image small (the default wheel bundles ~2 GB of CUDA)
RUN pip install --no-cache-dir --user torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu \\
    && pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \\
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
"""


def build(dest: Path) -> None:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    for f in files:
        if f.startswith("notebooks/"):  # big executed notebooks aren't needed to run the app
            continue
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    (dest / "Dockerfile").write_text(DOCKERFILE)
    header = "\n".join(["---", f"title: {SPACE['title']}", f"emoji: {SPACE['emoji']}",
                        f"colorFrom: {SPACE['from']}", f"colorTo: {SPACE['to']}", "sdk: docker", "app_port: 7860",
                        "pinned: false", f"short_description: {SPACE['description']}", "---", ""])
    readme = (ROOT / "README.md").read_text()
    # Space pages can't show repo-relative images unless they're uploaded too (they are), so keep the README as is.
    (dest / "README.md").write_text(header + "\n" + readme)


def main():
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("⚠️  Uncommitted changes won't be included (the Space is built from committed files):\n" + dirty)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=SPACE["name"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        build(dest)
        if args.dry_run:
            keep = ROOT / "outputs" / "space_build"
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(dest, keep)
            print(f"Built the Space folder at {keep} (not uploaded).")
            return
        from huggingface_hub import HfApi, whoami

        user = whoami()["name"]
        repo_id = f"{user}/{args.name}"
        api = HfApi()
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(dest),
                          commit_message="Deploy from GitHub repo")
        print(f"🚀 https://huggingface.co/spaces/{repo_id}  (first build takes ~5–10 minutes)")


if __name__ == "__main__":
    main()
