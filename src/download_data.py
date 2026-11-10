"""下载、校验并解压 CCTSDB2021 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

ZENODO_RECORD_ID = "17152727"
ARCHIVE_NAME = "dataset.rar"
DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ARCHIVE_NAME}?download=1"
)
ARCHIVE_SIZE = 1_555_591_303
ARCHIVE_MD5 = "f73e86e749c64cedc06dd7c538cdd359"

DEFAULT_RAW_DIR = "data/raw/CCTSDB2021"
DEFAULT_CACHE_DIR = "data/downloads"
CHUNK = 1 << 20

RETRY_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    OSError,
)


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0 or unit == "GB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} GB"


def log(msg: str) -> None:
    print(msg, flush=True)


ANNOTATION_DIRS = {
    "labels",
    "label",
    "labeltxt",
    "annotations",
    "annotation",
    "xmls",
    "xml",
}


def is_annotation_file(path: str | os.PathLike) -> bool:
    """判断路径是否像标注文件：xml 一律算，txt 须位于标注目录下。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".xml":
        return True
    return suffix == ".txt" and p.parent.name.lower() in ANNOTATION_DIRS


def count_annotations(root: str | os.PathLike, limit: int | None = None) -> int:
    """统计 root 下的标注文件数量，limit 用于只做快速探测。"""
    path = Path(root)
    if not path.is_dir():
        return 0
    found = 0
    for entry in path.rglob("*"):
        if is_annotation_file(entry) and entry.is_file():
            found += 1
            if limit is not None and found >= limit:
                break
    return found


def has_annotations(root: str | os.PathLike) -> bool:
    """判断 root 下是否已存在至少一个标注文件。"""
    return count_annotations(root, limit=1) > 0


def md5_of(path: str | os.PathLike, chunk: int = 4 * CHUNK) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_is_valid(path: str | os.PathLike, verify_md5: bool = True) -> bool:
    p = Path(path)
    if not p.is_file() or p.stat().st_size != ARCHIVE_SIZE:
        return False
    return not verify_md5 or md5_of(p) == ARCHIVE_MD5


class _Progress:
    def __init__(self, total: int | None, initial: int = 0, desc: str = "download") -> None:
        self.total = total
        self.done = initial
        self.desc = desc
        self._last = 0.0
        self._wrote = False
        self._bar = None
        if _tqdm is not None:
            self._bar = _tqdm(
                total=total,
                initial=initial,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=True,
            )

    def update(self, n: int) -> None:
        self.done += n
        if self._bar is not None:
            self._bar.update(n)
            return
        now = time.time()
        if now - self._last < 1.0:
            return
        self._last = now
        self._wrote = True
        if self.total:
            pct = 100.0 * self.done / self.total
            sys.stderr.write(
                f"\r{self.desc}: {pct:5.1f}%  "
                f"{human_bytes(self.done)} / {human_bytes(self.total)}   "
            )
        else:
            sys.stderr.write(f"\r{self.desc}: {human_bytes(self.done)}   ")
        sys.stderr.flush()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            return
        if self._wrote:
            sys.stderr.write("\n")
            sys.stderr.flush()


def download_file(
    url: str,
    dest: str | os.PathLike,
    *,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    retries: int = 5,
    timeout: float = 60.0,
    progress: bool = True,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if expected_size and part.exists() and part.stat().st_size > expected_size:
        part.unlink()  # 残留的旧版本分片，大小超预期就丢弃
    if dest.exists() and (not expected_size or dest.stat().st_size == expected_size):
        if not expected_md5 or md5_of(dest) == expected_md5:
            log(f"[download] already complete: {dest}")
            return dest
        log("[download] cached file failed the checksum, re-downloading")
        dest.unlink()

    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if expected_size and offset >= expected_size:
            break
        if offset:
            log(f"[download] resuming at {human_bytes(offset)} (attempt {attempt}/{retries})")
        req = urllib.request.Request(url)
        if offset:
            req.add_header("Range", f"bytes={offset}-")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if status != 206 and offset:
                    # 服务端忽略 Range 头，只能从头下载
                    offset = 0
                    part.unlink(missing_ok=True)
                length = resp.headers.get("Content-Length")
                total = expected_size or (offset + int(length) if length else None)
                mode = "ab" if offset else "wb"
                bar = _Progress(total, offset, "download") if progress else None
                with open(part, mode) as f:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        if bar is not None:
                            bar.update(len(block))
                if bar is not None:
                    bar.close()
            if expected_size and part.stat().st_size != expected_size:
                raise IOError(
                    f"incomplete file: {human_bytes(part.stat().st_size)} of "
                    f"{human_bytes(expected_size)}"
                )
            break
        except RETRY_ERRORS as exc:
            last_error = exc
            if attempt >= retries:
                break
            backoff = min(30.0, 2.0**attempt)
            log(f"[download] attempt {attempt}/{retries} failed ({exc}); retry in {backoff:.0f}s")
            time.sleep(backoff)

    if not part.exists():
        raise RuntimeError(f"download failed: {last_error}")
    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError(
            f"download incomplete after {retries} attempts "
            f"({human_bytes(part.stat().st_size)} / {human_bytes(expected_size)}); "
            f"run again to resume"
        )

    if expected_md5:
        log("[download] verifying MD5 ...")
        got = md5_of(part)
        if got != expected_md5:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch: expected {expected_md5}, got {got}; "
                f"the file was deleted, run again to re-download"
            )
        log("[download] MD5 ok")
    part.replace(dest)
    return dest


def _which_any(names: list[str], extra_paths: list[str] | None = None) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for raw in extra_paths or []:
        if Path(raw).is_file():
            return raw
    return None


def _extract_commands(archive: Path, dest: Path) -> list[tuple[str, list[str]]]:
    """按本机实际可用的解压工具生成候选命令列表。"""
    commands: list[tuple[str, list[str]]] = []
    if archive.suffix.lower() == ".zip":
        return commands

    seven = _which_any(
        ["7z", "7za", "7zz"],
        [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ],
    )
    unrar = _which_any(["unrar"])
    winrar = _which_any(
        ["WinRAR"],
        [
            r"C:\Program Files\WinRAR\WinRAR.exe",
            r"C:\Program Files (x86)\WinRAR\WinRAR.exe",
        ],
    )
    tar = _which_any(["bsdtar"], [r"C:\Windows\System32\tar.exe"]) or _which_any(["tar"])

    if unrar:
        commands.append(("unrar", [unrar, "x", "-o+", "-y", str(archive), str(dest) + os.sep]))
    if seven:
        commands.append(("7z", [seven, "x", "-y", f"-o{dest}", str(archive)]))
    if winrar:
        commands.append(("WinRAR", [winrar, "x", "-ibck", "-y", str(archive), str(dest) + os.sep]))
    if tar:
        commands.append(("tar", [tar, "-xf", str(archive), "-C", str(dest)]))
    return commands


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > limit:
                f.seek(size - limit)
            return f.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _run_command(
    cmd: list[str], log_dir: Path | None = None, timeout: float = 3600.0
) -> tuple[int, str]:

    base = Path(log_dir) if log_dir else Path(tempfile.gettempdir())
    work = base / f".cctsdb-extract-{os.getpid()}-{time.time_ns() % 1_000_000_000}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        out, err = work / "out.log", work / "err.log"
        with open(out, "wb") as fo, open(err, "wb") as fe:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=fo,
                stderr=fe,
                timeout=timeout,
            )
        tail = _tail(err) or _tail(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    lines = tail.splitlines()
    return proc.returncode, (lines[-1] if lines else "")


def extract_archive(archive: str | os.PathLike, dest: str | os.PathLike) -> str:
    archive = Path(archive)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(archive)

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return "zipfile"

    try:
        import rarfile  # type: ignore
    except Exception:
        rarfile = None

    if rarfile is not None:
        try:
            with rarfile.RarFile(str(archive)) as rf:
                rf.extractall(dest)
            return "rarfile"
        except Exception as exc:
            log(f"[extract] rarfile backend failed ({exc}); trying external tools")

    commands = _extract_commands(archive, dest)
    if not commands:
        raise RuntimeError(
            "no tool to unpack .rar found.\n"
            "  install one of: 7-Zip, WinRAR, unrar (`pip install rarfile` needs one of them)\n"
            "  then extract manually:\n"
            f"    tar -xf {archive} -C {dest}"
        )

    errors: list[str] = []
    for name, cmd in commands:
        log(f"[extract] trying {name}: {' '.join(cmd)}")
        try:
            code, tail = _run_command(cmd, log_dir=dest.parent)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if code == 0:
            return name
        errors.append(f"{name}: exit {code}" + (f" {tail}" if tail else ""))

    raise RuntimeError(
        "failed to extract the archive:\n  "
        + "\n  ".join(errors)
        + f"\n  extract it manually, then re-run:\n"
        f"    tar -xf {archive} -C {dest}\n"
        f"    (or: 7z x -y -o{dest} {archive})"
    )


def ensure_dataset(
    raw_dir: str | os.PathLike = DEFAULT_RAW_DIR,
    *,
    cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
    url: str = DOWNLOAD_URL,
    force: bool = False,
    verify: bool = True,
    keep_archive: bool = False,
    retries: int = 5,
) -> Path:
    raw = Path(raw_dir)
    if not force and has_annotations(raw):
        log(f"[data] raw dataset already present: {raw} (skip download)")
        return raw

    cache = Path(cache_dir)
    archive = cache / ARCHIVE_NAME
    if force:
        archive.unlink(missing_ok=True)

    if archive.is_file() and not archive_is_valid(archive, verify_md5=verify):
        log("[data] cached archive is incomplete or corrupt, removing it")
        archive.unlink()
        (cache / f"{ARCHIVE_NAME}.part").unlink(missing_ok=True)

    if not archive.is_file():
        log(f"[data] dataset not found, downloading {ARCHIVE_NAME} (~{human_bytes(ARCHIVE_SIZE)}) from Zenodo")
        log(f"[data]   {url}")
        log(f"[data]   cached at {archive} (resumable: just re-run after Ctrl+C)")
        download_file(
            url,
            archive,
            expected_size=ARCHIVE_SIZE,
            expected_md5=ARCHIVE_MD5 if verify else None,
            retries=retries,
        )
        log(f"[data] downloaded: {human_bytes(archive.stat().st_size)}")

    log(f"[data] extracting {archive.name} -> {raw}")
    tool = extract_archive(archive, raw)
    found = count_annotations(raw, limit=200_000)
    log(f"[data] extracted with {tool}: {found} annotation files under {raw}")
    if found == 0:
        raise RuntimeError(
            f"extraction finished but no annotation files found under {raw}.\n"
            f"  - check the archive manually: {archive}\n"
            f"  - then point --raw-dir at the folder that holds the label files"
        )

    if not keep_archive:
        archive.unlink(missing_ok=True)
        for leftover in (cache / f"{ARCHIVE_NAME}.part",):
            leftover.unlink(missing_ok=True)
        try:
            cache.rmdir()  # 目录非空时会失败，忽略即可
        except OSError:
            pass
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract CCTSDB2021 (called automatically by src.prepare_data).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="where the dataset is extracted")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="where dataset.rar is kept")
    parser.add_argument("--url", default=DOWNLOAD_URL, help="override the download URL (mirror)")
    parser.add_argument("--force", action="store_true", help="re-download even if data exists")
    parser.add_argument("--keep-archive", action="store_true", help="keep dataset.rar after extraction")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="skip the MD5 check")
    parser.add_argument("--retries", type=int, default=5, help="download retry count")
    parser.add_argument("--check", action="store_true", help="only report the current state")
    parser.add_argument("--prepare", action="store_true", help="run src.prepare_data afterwards")
    parser.add_argument("--prepare-out", default="data/processed/cctsdb", help="output dir for --prepare")
    return parser.parse_args(argv)


def report_state(raw_dir: str | os.PathLike, cache_dir: str | os.PathLike) -> None:
    raw = Path(raw_dir)
    cache = Path(cache_dir) / ARCHIVE_NAME
    log(f"raw dir      : {raw} -> {'found' if has_annotations(raw) else 'missing'} "
        f"({count_annotations(raw, limit=200_000)} annotations)")
    if cache.is_file():
        log(f"archive      : {cache} -> {human_bytes(cache.stat().st_size)} "
            f"(expected {human_bytes(ARCHIVE_SIZE)})")
    else:
        log(f"archive      : {cache} -> not cached")
    commands = _extract_commands(Path("probe.rar"), Path("probe"))
    tools = ", ".join(name for name, _ in commands) or "none"
    try:
        import rarfile  # noqa: F401

        tools = f"rarfile, {tools}" if tools != "none" else "rarfile"
    except Exception:
        pass
    log(f"extract tools: {tools}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check:
        report_state(args.raw_dir, args.cache_dir)
        return

    ensure_dataset(
        args.raw_dir,
        cache_dir=args.cache_dir,
        url=args.url,
        force=args.force,
        verify=args.verify,
        keep_archive=args.keep_archive,
        retries=args.retries,
    )
    if args.prepare:
        cmd = [
            sys.executable,
            "-m",
            "src.prepare_data",
            "--raw-dir",
            str(args.raw_dir),
            "--out-dir",
            args.prepare_out,
        ]
        log(f"[data] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
