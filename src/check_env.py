from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIN_PYTHON = (3, 10)
MIN_TORCH = (2, 0)
MIN_TORCHVISION = (0, 15)
DATA_NEED_GB = 8.0  

@dataclass
class Result:
    name: str
    status: str  
    detail: str = ""
    hint: str = ""

    def line(self) -> str:
        tag = {"ok": " OK ", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}[self.status]
        text = f"[{tag}] {self.name:<22} {self.detail}".rstrip()
        if self.status in {"warn", "fail"} and self.hint:
            text += f"\n         -> {self.hint}"
        return text


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.results.append(Result(name, status, detail, hint))

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def warned(self) -> list[Result]:
        return [r for r in self.results if r.status == "warn"]


def _version_tuple(text: str) -> tuple[int, ...]:
    nums: list[int] = []
    for part in text.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    return tuple(nums)


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def check_python(report: Report) -> None:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] >= MIN_PYTHON:
        report.add("python", "ok", f"{version} ({sys.executable})")
    else:
        report.add(
            "python", "fail", version,
            f"need >= {'.'.join(map(str, MIN_PYTHON))}; create the conda env from README",
        )
    report.info["python"] = version
    report.info["platform"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    report.info["cpu_count"] = os.cpu_count()
    report.add("platform", "ok", f"{report.info['platform']}, {os.cpu_count()} CPU cores")


def check_packages(report: Report) -> None:
    # 模块名 -> (pip 包名, 最低版本)
    required = {
        "numpy": ("numpy", "1.24"),
        "PIL": ("Pillow", "9.5"),
        "cv2": ("opencv-python", "4.8"),
        "matplotlib": ("matplotlib", "3.7"),
        "yaml": ("PyYAML", "6.0"),
    }
    optional = {"tqdm": "progress bars", "rarfile": "unpacking dataset.rar"}

    for module, (pip_name, min_version) in required.items():
        version = _module_version(module)
        if version is None:
            report.add(module, "fail", "not installed", f"pip install {pip_name}>={min_version}")
            continue
        report.info[module] = version
        if version != "unknown" and _version_tuple(version) < _version_tuple(min_version):
            report.add(
                module, "warn", f"{version} (need >= {min_version})",
                f"pip install -U {pip_name}>={min_version}",
            )
        else:
            report.add(module, "ok", version)

    for module, purpose in optional.items():
        version = _module_version(module)
        if version is None:
            report.add(module, "skip", f"not installed (optional: {purpose})")
        else:
            report.add(module, "ok", version)
            report.info[module] = version


def check_torch(report: Report) -> None:
    try:
        import torch
    except Exception as exc:  
        report.add("torch", "fail", f"import failed: {exc}", "pip install -r requirements.txt")
        return

    version = torch.__version__
    if _version_tuple(version) >= MIN_TORCH:
        report.add("torch", "ok", version)
    else:
        report.add("torch", "fail", version, f"need >= {'.'.join(map(str, MIN_TORCH))}")
    report.info["torch"] = version
    report.info["cuda_build"] = torch.version.cuda

    vision = _module_version("torchvision")
    if vision is None:
        report.add("torchvision", "fail", "not installed", "pip install torchvision")
    elif _version_tuple(vision) >= MIN_TORCHVISION:
        report.add("torchvision", "ok", vision)
        report.info["torchvision"] = vision
    else:
        report.add(
            "torchvision", "fail", vision, f"need >= {'.'.join(map(str, MIN_TORCHVISION))}"
        )

    try:  
        from torchvision.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        keep = nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
        report.add("torchvision ops", "ok", f"nms returns {len(keep)} box(es)")
    except Exception as exc:  # pragma: no cover
        report.add(
            "torchvision ops", "fail", str(exc)[:80],
            "torch/torchvision versions are incompatible; reinstall both together",
        )

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        vram = props.total_memory / 1024**3
        detail = f"CUDA {torch.version.cuda}, {props.name}, {vram:.1f} GB"
        report.info["gpu"] = props.name
        report.info["vram_gb"] = round(vram, 1)
        if vram >= 7.5:
            report.add("cuda", "ok", detail)
        else:
            report.add(
                "cuda", "warn", detail,
                "less than 8 GB: use --batch-size 1 and --min-size 448 --max-size 576",
            )
        try:
            bf16 = torch.cuda.is_bf16_supported()
            report.add("bf16", "ok" if bf16 else "skip", "supported" if bf16 else "not supported")
        except Exception:
            pass
    else:
        report.add(
            "cuda", "warn", "not available (CPU only)",
            "training will be very slow; install a CUDA build of torch or use a GPU machine",
        )


def check_disk(report: Report) -> None:
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / 1024**3
        report.info["disk_free_gb"] = round(free_gb, 1)
        if free_gb >= DATA_NEED_GB:
            report.add("disk space", "ok", f"{free_gb:.1f} GB free")
        elif free_gb >= DATA_NEED_GB / 2:
            report.add(
                "disk space", "warn", f"{free_gb:.1f} GB free",
                f"dataset + processed copy needs about {DATA_NEED_GB:.0f} GB",
            )
        else:
            report.add(
                "disk space", "fail", f"{free_gb:.1f} GB free",
                f"need at least {DATA_NEED_GB / 2:.0f} GB for data/raw + data/processed",
            )
    except OSError as exc:
        report.add("disk space", "warn", str(exc)[:80])


def check_write_access(report: Report) -> None:
    """检查 outputs/ 等输出目录是否可写。"""
    def probe(target: Path) -> str:
        target.mkdir(parents=True, exist_ok=True)
        name = f"write_test_{os.getpid()}.tmp"
        path = target / name
        path.write_text("ok", encoding="utf-8")
        path.unlink()
        return str(target)

    try:
        report.add("write access", "ok", f"{probe(Path('outputs'))}/ is writable")
        return
    except OSError as exc:
        outputs_error = str(exc)

    for fallback in (Path("."), Path("data")):
        try:
            where = probe(fallback)
        except OSError:
            continue
        report.add(
            "write access", "warn", f"outputs/ is not writable, but {where}/ is",
            f"{outputs_error}; training saves checkpoints under outputs/",
        )
        return

    report.add(
        "write access", "fail", outputs_error[:100],
        "run the commands from the project root and make sure the folder is not read-only",
    )


def check_extractor(report: Report) -> None:
    try:
        from .download_data import _extract_commands

        commands = _extract_commands(Path("dataset.rar"), Path("raw"))
        names = [name for name, _ in commands]
    except Exception as exc:  # pragma: no cover
        report.add("extractor", "warn", f"could not probe: {exc}")
        return

    if _module_version("rarfile"):
        names.insert(0, "rarfile")
    if names:
        report.add("extractor", "ok", ", ".join(names))
    else:
        report.add(
            "extractor", "warn", "no tool to unpack .rar",
            "install 7-Zip/WinRAR (Windows 10+ already has tar.exe); "
            "the download step will print a manual command otherwise",
        )


def check_config(report: Report) -> None:
    path = Path("configs/cctsdb_retinanet.yaml")
    if not path.is_file():
        report.add("config", "fail", f"missing {path}", "run from the project root")
        return
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sections = [k for k in ("data", "model", "train") if k in cfg]
        report.add("config", "ok", f"{path} ({', '.join(sections)})")
    except Exception as exc:
        report.add("config", "fail", str(exc)[:80])


def check_dataset(report: Report, raw_dir: Path, processed_dir: Path) -> None:
    try:
        from .download_data import count_annotations, has_annotations, human_bytes

        if has_annotations(raw_dir):
            found = count_annotations(raw_dir, limit=200_000)
            report.add("raw dataset", "ok", f"{raw_dir} ({found} annotation files)")
        else:
            report.add(
                "raw dataset", "warn", f"{raw_dir} has no annotation files",
                "prepare_data downloads it automatically on first run",
            )
    except Exception as exc:  # pragma: no cover
        report.add("raw dataset", "skip", f"probe failed: {exc}")

    images = processed_dir / "images"
    splits = {s: len(list((images / s).glob("*"))) for s in ("train", "val", "test")}
    report.info["processed_splits"] = splits
    if not any(splits.values()):
        report.add(
            "converted dataset", "warn", f"{processed_dir} not built",
            "python -m src.prepare_data --raw-dir data/raw/CCTSDB2021 --out-dir data/processed/cctsdb",
        )
        return
    bad = [s for s in ("train", "val") if splits[s] == 0]
    if bad:
        report.add("converted dataset", "fail", f"empty split(s): {', '.join(bad)}", "re-run prepare_data")
    else:
        report.add(
            "converted dataset", "ok",
            ", ".join(f"{k}={v}" for k, v in splits.items()),
        )
    try:
        used = sum(
            p.stat().st_size for p in (processed_dir / "images").rglob("*") if p.is_file()
        )
        report.info["processed_size_gb"] = round(used / 1024**3, 2)
    except OSError:
        pass


def run_checks(
    *,
    raw_dir: Path,
    processed_dir: Path,
    with_dataset: bool = True,
) -> Report:
    report = Report()
    check_python(report)
    check_packages(report)
    check_torch(report)
    check_disk(report)
    check_write_access(report)
    check_extractor(report)
    check_config(report)
    if with_dataset:
        check_dataset(report, raw_dir, processed_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the Python/PyTorch environment and dataset before training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="data/raw/CCTSDB2021")
    parser.add_argument("--processed-dir", default="data/processed/cctsdb")
    parser.add_argument("--no-dataset", dest="with_dataset", action="store_false", help="skip dataset checks")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", dest="json_path", default=None, help="also write a JSON report here")
    args = parser.parse_args(argv)

    report = run_checks(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        with_dataset=args.with_dataset,
    )

    print("=" * 78)
    print("environment check")
    print("=" * 78)
    for result in report.results:
        print(result.line())
    print("-" * 78)

    failed, warned = report.failed, report.warned
    print(f"{len(report.results)} checks: {len(failed)} failed, {len(warned)} warning(s)")
    if failed:
        print("\nblocking problems (fix these first):")
        for result in failed:
            print(f"  - {result.name}: {result.detail}")
    elif warned:
        print("\nwarnings are not fatal, but read the hints above.")

    if args.json_path:
        payload = {
            "results": [asdict(r) for r in report.results],
            "info": report.info,
            "ok": not failed and not (args.strict and warned),
        }
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport written to {path}")

    return 1 if failed or (args.strict and warned) else 0


if __name__ == "__main__":
    raise SystemExit(main())
