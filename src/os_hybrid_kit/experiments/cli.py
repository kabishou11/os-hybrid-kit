"""CLI for the compact offline fusion experiment module."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from os_hybrid_kit.experiments.experiment import (
    default_configs,
    format_results,
    format_sweep,
    load_fixture,
    run_config,
    sweep_configs,
)

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "examples" / "fusion_experiments"
DEFAULT_FIXTURES = (
    FIXTURE_DIR / "synthetic_win.json",
    FIXTURE_DIR / "synthetic_loss.json",
)
FOOTNOTE = (
    "Synthetic fixtures only: mathematical correctness and boundary behavior; "
    "no real retrieval gain is claimed. Eq. (3) alpha/lambda/kappa values from "
    "SciFact are dataset-specific named examples, not defaults. RRF and weighted "
    "rows replay in-tree FusionMethod math offline; no OpenSearch call. Ties break "
    "by descending doc id; epsilon=1e-8 stabilizes Eq. (1)."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m os_hybrid_kit.experiments",
        description="Replay fixed synthetic run lists through offline fusion and print metrics.",
    )
    parser.add_argument(
        "--fixture",
        nargs="*",
        type=Path,
        default=None,
        help="JSON fixture path(s); defaults to the bundled synthetic win/loss fixtures",
    )
    parser.add_argument("--no-sweep", action="store_true", help="skip the alpha/lambda/kappa grid")
    args = parser.parse_args(argv)

    for path in args.fixture or DEFAULT_FIXTURES:
        fixture = load_fixture(path)
        results = [run_config(fixture, config) for config in default_configs()]
        print(format_results(fixture, results))
        print()
        if fixture.sweep and not args.no_sweep:
            by_name = {result.config.name: result for result in results}
            best_single = max(
                by_name["lexical_only"].metrics.mean_ndcg,
                by_name["dense_only"].metrics.mean_ndcg,
            )
            sweep_results = [run_config(fixture, config) for config in sweep_configs()]
            print(format_sweep(fixture, sweep_results, by_name["rrf_equal"], best_single))
            print()
    print(FOOTNOTE)
    return 0
