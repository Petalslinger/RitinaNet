"""把多份评估结果汇总成 Markdown 对比表格。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = ["实验", "mAP@0.5", "mAP@0.5:0.95", "P", "R", "F1", "FPS", "图片数"]
METRIC_KEYS = ("mAP50", "mAP50_95", "precision", "recall", "f1")
DIGITS = 4


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        merged = dict(data["metrics"])
        merged.setdefault("fps", data.get("fps"))
        merged.setdefault("num_images", data.get("num_images"))
        return merged
    return data


def pretty_label(name: str) -> str:
    label = name.strip()
    label = label.replace("baseline", "RetinaNet").replace("base", "RetinaNet")
    label = label.replace("full", "RetinaNet+MDA").replace("mda", "MDA")
    label = label.replace("_softnms", " · Soft-NMS").replace("_nms", " · NMS")
    label = label.replace("_", " ")
    return " ".join(label.split())


def cell(metrics: dict, key: str, digits: int = DIGITS) -> str:
    value = metrics.get(key)
    if value is None:
        return "—"
    try:
        return format(float(value), f".{digits}f")
    except (TypeError, ValueError):
        return str(value)


def row(path: Path, label: str | None = None) -> list[str]:
    metrics = load_metrics(path)
    name = label or path.stem
    count = metrics.get("num_images")
    return [
        pretty_label(name),
        cell(metrics, "mAP50"),
        cell(metrics, "mAP50_95"),
        cell(metrics, "precision"),
        cell(metrics, "recall"),
        cell(metrics, "f1"),
        cell(metrics, "fps", digits=2),
        "—" if count is None else str(int(count)),
    ]


def table(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for item in rows:
        lines.append("| " + " | ".join(item) + " |")
    return "\n".join(lines)


def parse_group(spec: str) -> tuple[str, list[Path]]:
    if "|" not in spec:
        raise SystemExit(f'--group 需要 "标题|文件1,文件2" 的格式: {spec}')
    title, files = spec.split("|", 1)
    paths = [Path(part.strip()) for part in files.split(",") if part.strip()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("找不到评估结果:\n  " + "\n  ".join(missing))
    return title.strip(), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="TITLE|FILE1,FILE2",
        help="一组表格：标题 + 逗号分隔的 JSON 路径，可重复",
    )
    parser.add_argument("--output", default="report/compare_tables.md")
    args = parser.parse_args()

    if not args.group:
        raise SystemExit('至少给一个 --group "标题|文件1,文件2"')

    blocks: list[str] = []
    for spec in args.group:
        title, paths = parse_group(spec)
        rows = [row(path) for path in paths]
        blocks.append(f"## {title}\n\n{table(rows)}")

    markdown = "\n\n".join(blocks) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"[compare] saved: {output}")


if __name__ == "__main__":
    main()
