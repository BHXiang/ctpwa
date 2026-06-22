#!/usr/bin/env python3
"""Generate PEP 503 simple index from all wheels in dist/.
Usage: python scripts/gen_index.py [--base-url https://example.com/ctpwa]

Output: pypi-index/  (ready to deploy to any static host: GitHub Pages, S3, etc.)
"""
import hashlib
import sys
from pathlib import Path

BASE_URL = ""
for arg in sys.argv[1:]:
    if arg.startswith("--base-url="):
        BASE_URL = arg.split("=", 1)[1].rstrip("/")

WHEEL_DIR = Path("dist")
OUTPUT_DIR = Path("pypi-index")

OUTPUT_DIR.mkdir(exist_ok=True)

packages = {}
for whl in sorted(WHEEL_DIR.glob("*.whl")):
    name = whl.name.split("-")[0]
    packages.setdefault(name, []).append(whl)

# Root index
with open(OUTPUT_DIR / "index.html", "w") as f:
    for pkg in sorted(packages):
        f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
print(f"Root: {OUTPUT_DIR}/index.html")

# Per-package index
for pkg, wheels in packages.items():
    pkg_dir = OUTPUT_DIR / pkg
    pkg_dir.mkdir(exist_ok=True)

    with open(pkg_dir / "index.html", "w") as f:
        f.write("<!DOCTYPE html>\n<html><body>\n")
        for whl in sorted(wheels):
            dest = pkg_dir / whl.name
            # Copy if not already there
            if not dest.exists() or dest.stat().st_size != whl.stat().st_size:
                dest.write_bytes(whl.read_bytes())

            # SHA256 from the file we just wrote
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            href = f"{BASE_URL}/{pkg}/{whl.name}" if BASE_URL else whl.name

            # Extract Python version from wheel metadata for data-requires-python
            py_ver = ""
            for part in whl.name.split("-"):
                if part.startswith("cp"):
                    py_ver = ' data-requires-python="&gt;=3.12"'
                    break

            f.write(
                f'<a href="{href}"'
                f' data-sha256="{sha}"{py_ver}>{whl.name}</a><br>\n'
            )
        f.write("</body></html>\n")

    print(f"Package: {pkg_dir}/index.html ({len(wheels)} wheels)")


# Print usage instructions
print("")
print("=== 部署指南 ===")
print("方法1: GitHub Pages")
print("  1. 将 pypi-index/ 内容推送到 gh-pages 分支")
print("  2. 用户安装: pip install ctpwa --extra-index-url https://USER.github.io/REPO/")
print("")
print("方法2: 任意 HTTP 服务器")
print(f"  1. 将 pypi-index/ 放到服务器上")
print(f"  2. 重新生成索引: python scripts/gen_index.py --base-url https://your.server.com/path")
print(f"  3. 用户安装: pip install ctpwa --extra-index-url https://your.server.com/path/")
