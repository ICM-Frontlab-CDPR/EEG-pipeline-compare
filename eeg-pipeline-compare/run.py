"""Entry point: compare two EEG derivatives."""

import csv
from pathlib import Path
from typing import Any

from _config import load_config
from _ios import load
from _logger import logger
from _metrics import compute_metrics
from _io_scan import scan_derivative

CONFIG_PATH = Path("/Users/hippolyte.dreyfus/Documents/eeg-qc/config.yaml")
OUT_DIR = Path("/Users/hippolyte.dreyfus/Documents/eeg-qc/outputs")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict. Lists are joined as semicolon-separated strings."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = ";".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def _aggregate_rows(rows: list[dict], method: str) -> dict[str, Any]:
    """Aggregate a list of flat metric dicts into one row. Skips error rows and non-numeric cols."""
    valid_rows = [r for r in rows if "error" not in r]
    if not valid_rows:
        return {}
    if method == "mean":
        agg: dict[str, Any] = {"n_files": len(valid_rows), "n_errors": len(rows) - len(valid_rows)}
        for key in (valid_rows[0] if valid_rows else []):
            if key == "file":
                continue
            values = [r[key] for r in valid_rows if key in r and r[key] is not None]
            numeric = []
            for v in values:
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    pass
            agg[key] = sum(numeric) / len(numeric) if numeric else None
        return agg
    raise ValueError(f"Unknown aggregation method: {method}")


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aggregated_rows: list[dict] = []

    for deriv in cfg.derivatives:  # TODO PARALLELIZATION POINT for later
        logger.info("Scanning derivative '%s' (%s)…", deriv.name, deriv.path)
        files = scan_derivative(deriv.path, bids=deriv.bids, proc=deriv.proc)
        logger.info("  → %d EEG file(s) found", len(files))

        rows: list[dict] = []
        for fpath in files:
            logger.info("  Extracting metrics: %s", fpath.name)
            try:
                data = load(str(fpath))
                metrics = compute_metrics(data)
                row = {"file": fpath.name}
                row.update(_flatten(metrics))
            except Exception as exc:
                logger.warning("  SKIP %s — %s", fpath.name, exc)
                row = {"file": fpath.name, "error": str(exc)}
            rows.append(row)

        # -- per-file CSV --
        out_file = OUT_DIR / f"{deriv.name}_metrics.csv"
        if rows:
            fieldnames = list(dict.fromkeys(k for row in rows for k in row))
            with out_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        logger.info("  Saved per-file → %s (%d files)", out_file, len(rows))

        # -- aggregation --
        agg = _aggregate_rows(rows, method=cfg.aggregation_type)
        agg["derivative"] = deriv.name
        aggregated_rows.append(agg)
        logger.info("  Aggregated (%s) → %d numeric metrics", cfg.aggregation_type, len(agg))

    # -- aggregated summary CSV (one row per derivative) --
    if aggregated_rows:
        fieldnames = ["derivative", "n_files", "n_errors"] + [
            k for k in aggregated_rows[0] if k not in ("derivative", "n_files", "n_errors")
        ]
        summary_file = OUT_DIR / "summary_aggregated.csv"
        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(aggregated_rows)
        logger.info("Summary saved → %s", summary_file)

