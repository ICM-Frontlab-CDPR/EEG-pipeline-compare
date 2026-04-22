"""Entry point: compare two EEG derivatives."""

import itertools
from pathlib import Path

from _config import load_config
from _ios import load
from _logger import logger
from _metrics import compute_metrics
from _scan import scan_derivative
from _viz import dual_derivative_figure as viz

CONFIG_PATH = Path("/Users/hippolyte.dreyfus/Documents/eeg-qc/config.yaml")
OUT_DIR = Path("/Users/hippolyte.dreyfus/Documents/eeg-qc/outputs")

if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)

    for deriv in cfg.derivatives:
        logger.info("Scanning derivative '%s' (%s)…", deriv.name, deriv.path)
        files = scan_derivative(deriv.path, bids=deriv.bids)
        logger.info("  → %d EEG file(s) found:", len(files))
        for f in files:
            logger.info("    %s", f)

