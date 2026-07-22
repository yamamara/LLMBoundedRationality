from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


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

    (output_dir / "cournot_player_scores.svg").write_text(
        _cournot_score_svg(scores),
        encoding="utf-8",
    )


def _cournot_score_svg(scores: list[CournotPlayerScore]) -> str:
    width = max(760, 180 + 120 * len(scores))
    height = 540
    left, right, top, bottom = 90, 40, 85, 125
    plot_width = width - left - right
    plot_height = height - top - bottom

    domain_min = min(0.0, *(score.lower_two_sd for score in scores))
    domain_max = max(0.0, *(score.upper_two_sd for score in scores))
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

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="chart-title chart-description">',
        '<title id="chart-title">Cournot average profit by player</title>',
        '<desc id="chart-description">Bars show average profit across all completed rounds. '
        'Whiskers show plus or minus two sample standard deviations.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#111827">Cournot Average Profit by Player</text>',
        f'<text x="{left}" y="58" font-family="sans-serif" font-size="13" '
        'fill="#4b5563">Bars: average profit across all rounds | Whiskers: +/- 2 SD</text>',
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
        title = escape(
            f"{score.run_id} / {score.player_id}: mean {score.average_profit:.2f}, "
            f"SD {score.standard_deviation:.2f}"
        )
        parts.extend(
            [
                f'<g data-run-id="{escape(score.run_id, quote=True)}" '
                f'data-player-id="{escape(score.player_id, quote=True)}">',
                f'<title>{title}</title>',
                f'<rect x="{center_x - bar_width / 2:.2f}" y="{bar_y:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>',
                f'<line x1="{center_x:.2f}" y1="{y(score.lower_two_sd):.2f}" '
                f'x2="{center_x:.2f}" y2="{y(score.upper_two_sd):.2f}" '
                'stroke="#111827" stroke-width="2"/>',
                f'<line x1="{center_x - 10:.2f}" y1="{y(score.lower_two_sd):.2f}" '
                f'x2="{center_x + 10:.2f}" y2="{y(score.lower_two_sd):.2f}" '
                'stroke="#111827" stroke-width="2"/>',
                f'<line x1="{center_x - 10:.2f}" y1="{y(score.upper_two_sd):.2f}" '
                f'x2="{center_x + 10:.2f}" y2="{y(score.upper_two_sd):.2f}" '
                'stroke="#111827" stroke-width="2"/>',
                f'<text x="{center_x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12" font-weight="700" '
                f'fill="#111827">{score.average_profit:.2f}</text>',
                f'<text x="{center_x:.2f}" y="{height - bottom + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12" font-weight="700" '
                f'fill="#111827">{escape(score.player_id)}</text>',
                f'<text x="{center_x:.2f}" y="{height - bottom + 42}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10" fill="#4b5563">{escape(score.agent)}</text>',
            ]
        )
        if multiple_runs:
            parts.append(
                f'<text x="{center_x:.2f}" y="{height - bottom + 58}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="9" fill="#6b7280">{escape(run_label)}</text>'
            )
        parts.append("</g>")

    legend_y = height - 26
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
            "</svg>\n",
        ]
    )
    return "\n".join(parts)

