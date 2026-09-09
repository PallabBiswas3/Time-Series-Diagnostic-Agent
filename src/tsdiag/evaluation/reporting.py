from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def write_json_report(result: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain(result), indent=2, default=str), encoding="utf-8")
    return path


def write_fault_table(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown_summary(summary: dict[str, Any], table_rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Tennessee Eastman benchmark summary", ""]
    lines.append("## Aggregate metrics")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Per-fault table")
    lines.append("")
    rows = list(table_rows)
    if rows:
        fields = list(rows[0].keys())
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_detection_plots(table_rows: Iterable[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Create lightweight publication-draft benchmark plots.

    Matplotlib is imported lazily so the core package remains usable in minimal
    environments where plotting is not needed.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in table_rows if int(r.get("fault_id", 0)) != 0]
    if not rows:
        return []

    fault_labels = [f"IDV({int(r['fault_id']):02d})" for r in rows]
    detection = [float(r.get("post_fault_detection_rate") or 0.0) for r in rows]
    false_alarm = [float(r.get("pre_fault_false_alarm_rate") or 0.0) for r in rows]
    delay = [float(r.get("detection_delay_samples") or 0.0) for r in rows]

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(rows))
    ax.bar(x, detection, label="Post-fault detection rate")
    ax.plot(list(x), false_alarm, marker="o", label="Pre-fault false-alarm rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fault_labels, rotation=60, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("TEP detection rate and false-alarm rate by fault")
    ax.legend()
    fig.tight_layout()
    p = output_dir / "tep_detection_false_alarm_by_fault.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(fault_labels, delay)
    ax.set_xticklabels(fault_labels, rotation=60, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("TEP first-detection delay by fault")
    fig.tight_layout()
    p = output_dir / "tep_detection_delay_by_fault.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    return paths
