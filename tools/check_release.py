"""Reject credentials, machine-specific paths and weight files in tracked content."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
credential = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
machine = re.compile(r'(?:/(?:Users|home)/[^/\s]+/|/var/lib/|192\.168\.\d+\.\d+|10(?:\.\d{1,3}){3}|100\.(?:6[4-9]|[789]\d|1[01]\d|12[0-7])\.\d+\.\d+)')
for name in filter(None, paths):
    path = root / name
    if path.is_symlink() or path.suffix in {'.safetensors', '.gguf', '.bin', '.pt', '.pth', '.key', '.pem'}:
        raise SystemExit('Unshippable tracked file: ' + name)
    if path.stat().st_size > 1024 * 1024:
        raise SystemExit('Unexpected large tracked file: ' + name)
    text = path.read_text()
    if credential.search(text) or (name != 'tools/check_release.py' and machine.search(text)):
        raise SystemExit('Private data pattern in: ' + name)
print('Tracked files contain no detected credentials, private machine paths or weight binaries.')
