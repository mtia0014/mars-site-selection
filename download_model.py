"""下载嵌入模型到本地 models/<repo_name>/ 目录（绕过 HF Hub 的 HEAD 元数据请求）

背景：hf-mirror.com 对 HEAD 请求会 308 重定向到 huggingface.co（本机 SSL 被墙），
而 GET 请求可正常下载。故直接用 requests GET 拉取全部必需文件到本地，供离线加载。

用法:
    python download_model.py BAAI/bge-m3
    python download_model.py                 # 默认 BAAI/bge-m3
"""

import os
import sys
import time

import requests

BASE = "https://hf-mirror.com"

# 这些文件对 sentence-transformers 推理无用，跳过
SKIP_PREFIX = (".gitattributes", "README", "imgs/", "onnx/")
SKIP_SUFFIX = (".jpg", ".webp", ".DS_Store")


def list_files(repo: str):
    url = f"{BASE}/api/models/{repo}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    names = [s.get("rfilename", "") for s in r.json().get("siblings", [])]
    has_safetensors = "model.safetensors" in names
    out = []
    for fn in names:
        if fn.startswith(SKIP_PREFIX) or fn.endswith(SKIP_SUFFIX):
            continue
        # 已有 safetensors 权重时，跳过冗余的 pytorch_model.bin（省 1GB 下载）
        if has_safetensors and fn == "pytorch_model.bin":
            continue
        out.append(fn)
    return out


def download(url, dest, retries=5):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=(15, 300)) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, dest)
            return True
        except Exception as e:
            print(f"    重试 {i + 1}/{retries}: {type(e).__name__} {str(e)[:120]}")
            time.sleep(3)
    return False


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-m3"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", repo.split("/")[-1])
    print(f"模型: {repo}\n目标目录: {out_dir}\n")

    files = list_files(repo)
    print(f"共 {len(files)} 个文件\n")

    failed = []
    for fn in files:
        dest = os.path.join(out_dir, fn.replace("/", os.sep))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"跳过（已存在）: {fn}")
            continue
        print(f"下载: {fn}")
        if not download(f"{BASE}/{repo}/resolve/main/{fn}", dest):
            print(f"❌ 下载失败: {fn}")
            failed.append(fn)
            continue
        print(f"  ✅ {fn} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")

    if failed:
        print(f"\n有 {len(failed)} 个文件失败，请重试: {failed}")
        sys.exit(1)
    print(f"\n全部完成: {out_dir}")


if __name__ == "__main__":
    main()
