from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from sandbox.provenance import visualization_provenance
from sandbox.statistics import t_critical_95


@dataclass(frozen=True)
class CournotPlayerScore:
    run_id: str
    player_id: str
    agent: str
    agent_type: str
    round_count: int
    average_profit: float
    standard_deviation: float
    lower_two_sd: float
    upper_two_sd: float
    ci95_half_width: float
    ci95_lower: float
    ci95_upper: float
    minimum_profit: float
    maximum_profit: float
    total_profit: float


def cournot_player_scores(run_dirs: Iterable[Path]) -> list[CournotPlayerScore]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for run_dir in run_dirs:
        with (run_dir / "participant_data.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["run_id"],
                    row["player_id"],
                    row["agent"],
                    row["agent_type"],
                )
                grouped[key].append(float(row["profit"]))

    scores = []
    for (run_id, player_id, agent, agent_type), profits in sorted(grouped.items()):
        average = mean(profits)
        deviation = stdev(profits) if len(profits) > 1 else 0.0
        ci95_half_width = (
            t_critical_95(len(profits) - 1) * deviation / math.sqrt(len(profits))
            if len(profits) > 1
            else 0.0
        )
        scores.append(
            CournotPlayerScore(
                run_id=run_id,
                player_id=player_id,
                agent=agent,
                agent_type=agent_type,
                round_count=len(profits),
                average_profit=average,
                standard_deviation=deviation,
                lower_two_sd=average - 2 * deviation,
                upper_two_sd=average + 2 * deviation,
                ci95_half_width=ci95_half_width,
                ci95_lower=average - ci95_half_width,
                ci95_upper=average + ci95_half_width,
                minimum_profit=min(profits),
                maximum_profit=max(profits),
                total_profit=sum(profits),
            )
        )
    return scores


def write_cournot_player_visualization(
    run_dirs: Iterable[Path],
    output_dir: Path,
) -> None:
    run_dirs = list(run_dirs)
    scores = cournot_player_scores(run_dirs)
    if not scores:
        raise ValueError("Cournot visualization requires participant profit rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cournot_player_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(asdict(scores[0]))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(score) for score in scores)

    provenance = visualization_provenance(run_dirs)
    provenance_json = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    short_hash = provenance["short_parameter_hash"]
    (output_dir / f"cournot_parameters_{short_hash}.json").write_text(
        provenance_json, encoding="utf-8"
    )
    (output_dir / "cournot_parameters.json").write_text(
        provenance_json, encoding="utf-8"
    )

    two_sd_svg = cournot_score_svg(scores, "two_sd", provenance)
    ci95_svg = cournot_score_svg(scores, "ci95", provenance)
    for filename in (
        "cournot_player_scores.svg",
        provenance["images"]["two_standard_deviations"],
    ):
        (output_dir / filename).write_text(two_sd_svg, encoding="utf-8")
    for filename in (
        "cournot_player_scores_ci95.svg",
        provenance["images"]["student_t_95_ci"],
    ):
        (output_dir / filename).write_text(ci95_svg, encoding="utf-8")


def cournot_score_svg(
    scores: list[CournotPlayerScore],
    error_mode: str = "two_sd",
    provenance: dict[str, Any] | None = None,
    sample_description: str = "all completed rounds",
    show_whiskers: bool = True,
) -> str:
    if error_mode == "two_sd":
        lower_bound = lambda score: score.lower_two_sd
        upper_bound = lambda score: score.upper_two_sd
        whisker_description = "plus or minus two sample standard deviations"
        whisker_label = "+/- 2 SD"
    elif error_mode == "ci95":
        lower_bound = lambda score: score.ci95_lower
        upper_bound = lambda score: score.ci95_upper
        whisker_description = "95% Student-t confidence intervals"
        whisker_label = "95% t confidence interval"
    else:
        raise ValueError(f"Unsupported Cournot error mode: {error_mode}")

    width = max(760, 180 + 120 * len(scores))
    chart_height = 540
    provenance = provenance or {}
    experiment_code = provenance.get("parameter_code", "")
    code_line_length = max(80, (width - 80) // 6)
    code_lines = [
        experiment_code[index : index + code_line_length]
        for index in range(0, len(experiment_code), code_line_length)
    ]
    code_section_height = 42 + 9 * len(code_lines) if code_lines else 0
    height = chart_height + code_section_height
    left, right, top, bottom = 90, 40, 105, 125
    plot_width = width - left - right
    plot_height = chart_height - top - bottom

    domain_min = min(0.0, *(lower_bound(score) for score in scores))
    domain_max = max(0.0, *(upper_bound(score) for score in scores))
    span = domain_max - domain_min
    if span == 0:
        domain_min, domain_max = -1.0, 1.0
    else:
        padding = span * 0.08
        domain_min -= padding
        domain_max += padding
    span = domain_max - domain_min

    def y(value: float) -> float:
        return top + (domain_max - value) / span * plot_height

    zero_y = y(0.0)
    slot_width = plot_width / len(scores)
    bar_width = min(64.0, slot_width * 0.56)
    multiple_runs = len({score.run_id for score in scores}) > 1

    full_hash = provenance.get("parameter_hash", "unknown")
    short_hash = provenance.get("short_parameter_hash", full_hash[:16])
    metadata = escape(json.dumps(provenance, sort_keys=True))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'data-parameter-hash="{escape(full_hash, quote=True)}" '
        f'data-parameter-code="{experiment_code}" '
        'aria-labelledby="chart-title chart-description">',
        '<title id="chart-title">Cournot average profit by player</title>',
        f'<desc id="chart-description">Bars summarize {escape(sample_description)}. Whiskers show '
        f'{whisker_description if show_whiskers else "no interval because fewer than two samples are available"}. '
        'The numeric experiment code decodes '
        'to the game and player parameters.</desc>',
        f'<metadata id="cournot-experiment-provenance">{metadata}</metadata>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#111827">Cournot Average Profit by Player</text>',
        f'<text x="{left}" y="58" font-family="sans-serif" font-size="13" '
        f'fill="#4b5563">Bars: {escape(sample_description)} | Whiskers: '
        f'{whisker_label if show_whiskers else "unavailable (n &lt; 2)"}</text>',
        f'<text x="{left}" y="79" font-family="monospace" font-size="11" '
        f'fill="#6b7280">Experiment code v1 | integrity {escape(short_hash)}</text>',
    ]

    for index in range(6):
        value = domain_min + span * index / 5
        tick_y = y(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{tick_y:.2f}" x2="{width - right}" '
                f'y2="{tick_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11" fill="#6b7280">{value:.1f}</text>',
            ]
        )

    parts.append(
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" '
        f'y2="{zero_y:.2f}" stroke="#111827" stroke-width="1.5"/>'
    )

    for index, score in enumerate(scores):
        center_x = left + slot_width * (index + 0.5)
        mean_y = y(score.average_profit)
        bar_y = min(mean_y, zero_y)
        bar_height = abs(mean_y - zero_y)
        color = "#2563eb" if score.agent_type == "AI" else "#d97706"
        label_y = mean_y - 8 if score.average_profit >= 0 else mean_y + 18
        run_label = score.run_id if len(score.run_id) <= 18 else score.run_id[:15] + "..."
        interval = (
            f", {whisker_label} [{lower_bound(score):.2f}, {upper_bound(score):.2f}]"
            if show_whiskers
            else ""
        )
        title = escape(
            f"{score.run_id} / {score.player_id}: mean {score.average_profit:.2f}"
            f"{interval}"
        )
        parts.extend(
            [
                f'<g data-run-id="{escape(score.run_id, quote=True)}" '
                f'data-player-id="{escape(score.player_id, quote=True)}">',
                f'<title>{title}</title>',
                f'<rect x="{center_x - bar_width / 2:.2f}" y="{bar_y:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>',
            ]
        )
        if show_whiskers:
            parts.extend(
                [
                f'<line x1="{center_x:.2f}" y1="{y(lower_bound(score)):.2f}" '
                f'x2="{center_x:.2f}" y2="{y(upper_bound(score)):.2f}" '
                'stroke="#111827" stroke-width="2"/>',
                f'<line x1="{center_x - 10:.2f}" y1="{y(lower_bound(score)):.2f}" '
                f'x2="{center_x + 10:.2f}" y2="{y(lower_bound(score)):.2f}" '
                'stroke="#111827" stroke-width="2"/>',
                f'<line x1="{center_x - 10:.2f}" y1="{y(upper_bound(score)):.2f}" '
                f'x2="{center_x + 10:.2f}" y2="{y(upper_bound(score)):.2f}" '
                'stroke="#111827" stroke-width="2"/>',
                ]
            )
        parts.extend(
            [
                f'<text x="{center_x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12" font-weight="700" '
                f'fill="#111827">{score.average_profit:.2f}</text>',
                f'<text x="{center_x:.2f}" y="{chart_height - bottom + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12" font-weight="700" '
                f'fill="#111827">{escape(score.player_id)}</text>',
                f'<text x="{center_x:.2f}" y="{chart_height - bottom + 42}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10" fill="#4b5563">{escape(score.agent)}</text>',
            ]
        )
        if multiple_runs:
            parts.append(
                f'<text x="{center_x:.2f}" y="{chart_height - bottom + 58}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="9" fill="#6b7280">{escape(run_label)}</text>'
            )
        parts.append("</g>")

    legend_y = chart_height - 26
    parts.extend(
        [
            f'<rect x="{left}" y="{legend_y - 11}" width="12" height="12" fill="#2563eb"/>',
            f'<text x="{left + 18}" y="{legend_y}" font-family="sans-serif" '
            'font-size="11" fill="#374151">AI</text>',
            f'<rect x="{left + 60}" y="{legend_y - 11}" width="12" height="12" fill="#d97706"/>',
            f'<text x="{left + 78}" y="{legend_y}" font-family="sans-serif" '
            'font-size="11" fill="#374151">Human</text>',
            f'<text x="22" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
            'transform="rotate(-90 22 '
            f'{top + plot_height / 2:.2f})" font-family="sans-serif" font-size="12" '
            'fill="#374151">Average profit</text>',
        ]
    )
    if code_lines:
        parts.extend(
            [
                f'<line x1="40" y1="{chart_height}" x2="{width - 40}" '
                f'y2="{chart_height}" stroke="#d1d5db"/>',
                f'<text x="40" y="{chart_height + 20}" font-family="sans-serif" '
                'font-size="11" font-weight="700" fill="#374151">'
                'Numeric experiment code - decode with scripts/decode_cournot_code.py</text>',
                f'<g id="cournot-parameter-code" data-parameter-code="{experiment_code}">',
            ]
        )
        for index, line in enumerate(code_lines):
            parts.append(
                f'<text x="40" y="{chart_height + 36 + index * 9}" '
                f'font-family="monospace" font-size="7" fill="#4b5563">{line}</text>'
            )
        parts.append("</g>")
    parts.append("</svg>\n")
    return "\n".join(parts)
