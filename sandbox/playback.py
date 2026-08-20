from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable


def cournot_playback_data(run_dirs: Iterable[Path]) -> dict:
    return {"runs": [_cournot_run_playback(run_dir) for run_dir in run_dirs]}


def write_cournot_playback(run_dirs: Iterable[Path], output_dir: Path) -> None:
    data = cournot_playback_data(run_dirs)
    if not data["runs"] or any(not run["timeline"] for run in data["runs"]):
        raise ValueError("Cournot playback requires decision and settlement events")
    embedded = json.dumps(data, separators=(",", ":"), sort_keys=True)
    embedded = embedded.replace("</", "<\\/").replace("\u2028", "\\u2028").replace(
        "\u2029", "\\u2029"
    )
    html = _PLAYBACK_HTML.replace("__PLAYBACK_DATA__", embedded)
    (output_dir / "cournot_playback.html").write_text(html, encoding="utf-8")


def _cournot_run_playback(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    player_metadata: dict[str, dict[str, str]] = {}
    with (run_dir / "participant_data.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            player_metadata.setdefault(
                row["player_id"],
                {
                    "player_id": row["player_id"],
                    "agent": row["agent"],
                    "agent_type": row["agent_type"],
                },
            )

    players = {
        player_id: {
            **metadata,
            "quantity": None,
            "profit": None,
            "state": "pending",
            "active": False,
        }
        for player_id, metadata in player_metadata.items()
    }
    timeline = []
    current_round = None
    last_settlement = None

    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            event_type = event.get("event_type")
            if event_type not in {"decision", "settlement"}:
                continue
            round_number = int(event["round"])
            if round_number != current_round:
                current_round = round_number
                for player in players.values():
                    player["state"] = "pending"
                    player["active"] = False

            if event_type == "decision":
                player_id = event["player_id"]
                if player_id not in players:
                    players[player_id] = {
                        "player_id": player_id,
                        "agent": "unknown",
                        "agent_type": "unknown",
                        "quantity": None,
                        "profit": None,
                        "state": "pending",
                        "active": False,
                    }
                for player in players.values():
                    player["active"] = False
                applied_action = event.get("applied_action") or {}
                quantity = (applied_action.get("value") or {}).get("quantity")
                players[player_id]["quantity"] = quantity
                players[player_id]["state"] = (
                    "revised" if event.get("revision_allowed", True) else "held"
                )
                players[player_id]["active"] = True
                compact_event = _compact_decision_event(event)
                provisional = [
                    player["quantity"]
                    for player in players.values()
                    if player["quantity"] is not None
                ]
                market = {
                    "stage": "provisional",
                    "provisional_total_quantity": sum(provisional),
                    "price": None,
                    "total_market_profit": None,
                    "last_settlement": last_settlement,
                }
            else:
                for player in players.values():
                    player["active"] = False
                for result in event.get("firm_results", []):
                    player_id = result["player_id"]
                    if player_id not in players:
                        players[player_id] = {
                            "player_id": player_id,
                            "agent": "unknown",
                            "agent_type": "unknown",
                            "active": False,
                        }
                    players[player_id].update(
                        {
                            "quantity": result.get("quantity"),
                            "profit": result.get("profit"),
                            "state": "settled",
                            "active": False,
                        }
                    )
                last_settlement = {
                    "round": round_number,
                    "total_quantity": event.get("total_quantity"),
                    "price": event.get("price"),
                    "total_market_profit": event.get("total_market_profit"),
                }
                compact_event = {
                    "event_type": "settlement",
                    "round": round_number,
                    "firm_results": event.get("firm_results", []),
                    "total_quantity": event.get("total_quantity"),
                    "price": event.get("price"),
                    "total_market_profit": event.get("total_market_profit"),
                    "distance_to_nash": event.get("distance_to_nash"),
                }
                market = {"stage": "settled", **last_settlement}

            timeline.append(
                {
                    "index": len(timeline),
                    "round": round_number,
                    "event_type": event_type,
                    "treatment": event.get("treatment")
                    or (event.get("observation", {}).get("public_state", {}).get("treatment"))
                    or summary.get("cournot", {}).get("treatment", ""),
                    "players": deepcopy(list(players.values())),
                    "market": market,
                    "event": compact_event,
                }
            )

    return {
        "run_id": summary.get("run_id", run_dir.name),
        "game_name": summary.get("game_name", "Cournot experiment"),
        "players": list(player_metadata.values()),
        "timeline": timeline,
    }


def _compact_decision_event(event: dict) -> dict:
    metadata = event.get("agent_metadata") or {}
    private_state = event.get("observation", {}).get("private_state", {})
    decision_support = private_state.get("decision_support") or {}
    best_reply = private_state.get("best_reply_quantity")
    if best_reply is None:
        best_reply = decision_support.get("best_reply_quantity")
    returned_action = event.get("returned_action") or {}
    return {
        "event_type": "decision",
        "round": int(event["round"]),
        "player_id": event.get("player_id"),
        "revision_allowed": event.get("revision_allowed", True),
        "valid": event.get("valid"),
        "invalid_reason": event.get("invalid_reason"),
        "returned_action": returned_action,
        "applied_action": event.get("applied_action"),
        "reasoning": returned_action.get("reasoning", ""),
        "observation": event.get("observation"),
        "best_reply_quantity": best_reply,
        "metadata": {
            key: metadata[key]
            for key in (
                "forced_hold",
                "retry_count",
                "latency_seconds",
                "policy",
                "response_model",
                "usage",
            )
            if key in metadata
        },
    }


_PLAYBACK_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cournot Event Playback</title>
<style>
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #5f6878;
  --line: #d9dee7;
  --panel: #f6f7f9;
  --blue: #2563eb;
  --blue-soft: #e8f0ff;
  --amber: #b45309;
  --amber-soft: #fff3df;
  --green: #087f5b;
  --green-soft: #e6f6f0;
  --red: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #ffffff;
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  letter-spacing: 0;
}
button, select, input { font: inherit; }
button:focus-visible, select:focus-visible, input:focus-visible {
  outline: 3px solid rgba(37, 99, 235, 0.25);
  outline-offset: 2px;
}
.shell { max-width: 1180px; margin: 0 auto; padding: 28px 28px 56px; }
.topbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--line);
}
h1 { margin: 0 0 5px; font-size: 25px; line-height: 1.2; }
.subtitle { margin: 0; color: var(--muted); }
.run-picker { display: grid; gap: 6px; min-width: 260px; }
label { color: var(--muted); font-size: 12px; font-weight: 650; }
select {
  min-height: 38px;
  border: 1px solid #b8c0ce;
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  padding: 7px 34px 7px 10px;
}
.timeline { padding: 20px 0 18px; border-bottom: 1px solid var(--line); }
.timeline-meta { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.event-name { font-weight: 750; }
.event-count { color: var(--muted); font-variant-numeric: tabular-nums; }
.slider { width: 100%; accent-color: var(--blue); cursor: pointer; }
.controls { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
.icon-button {
  width: 38px;
  height: 38px;
  display: inline-grid;
  place-items: center;
  border: 1px solid #b8c0ce;
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  cursor: pointer;
}
.icon-button:hover { background: var(--panel); }
.icon-button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
.speed { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.speed select { min-height: 34px; padding-top: 4px; padding-bottom: 4px; }
.section { padding: 22px 0; border-bottom: 1px solid var(--line); }
.section-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 14px; }
h2 { margin: 0; font-size: 17px; }
.badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; }
.badge {
  display: inline-flex;
  align-items: center;
  min-height: 25px;
  border-radius: 999px;
  padding: 3px 9px;
  background: #edf0f5;
  color: #465065;
  font-size: 11px;
  font-weight: 750;
  text-transform: uppercase;
}
.badge.provisional { background: var(--amber-soft); color: var(--amber); }
.badge.settled { background: var(--green-soft); color: var(--green); }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
.metric { min-height: 76px; padding: 13px 15px; background: var(--panel); border-right: 1px solid var(--line); }
.metric:last-child { border-right: 0; }
.metric-name { display: block; margin-bottom: 7px; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
.metric-value { font-size: 19px; font-weight: 760; font-variant-numeric: tabular-nums; }
.table-wrap { margin-top: 16px; overflow-x: auto; border: 1px solid var(--line); border-radius: 7px; }
table { width: 100%; border-collapse: collapse; min-width: 690px; }
th, td { padding: 11px 13px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
th { background: var(--panel); color: var(--muted); font-size: 11px; text-transform: uppercase; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr.active { background: var(--blue-soft); box-shadow: inset 4px 0 var(--blue); }
.number { text-align: right; font-variant-numeric: tabular-nums; }
.player { font-weight: 760; }
.agent { color: var(--muted); font-size: 12px; }
.state { font-size: 12px; font-weight: 700; }
.state.revised { color: var(--blue); }
.state.held { color: var(--amber); }
.state.settled { color: var(--green); }
.inspector-grid { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.4fr); gap: 22px; }
.facts { display: grid; grid-template-columns: 140px minmax(0, 1fr); border-top: 1px solid var(--line); }
.fact-label, .fact-value { padding: 9px 0; border-bottom: 1px solid var(--line); }
.fact-label { color: var(--muted); font-size: 12px; }
.fact-value { overflow-wrap: anywhere; font-weight: 620; }
.reasoning { margin: 16px 0 0; padding: 13px 14px; border-left: 3px solid var(--blue); background: var(--panel); line-height: 1.5; }
.data-block { margin-bottom: 14px; }
.data-block:last-child { margin-bottom: 0; }
.data-block h3 { margin: 0 0 7px; font-size: 12px; color: var(--muted); text-transform: uppercase; }
pre {
  margin: 0;
  max-height: 300px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #111827;
  color: #e5e7eb;
  padding: 13px;
  font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
@media (max-width: 760px) {
  .shell { padding: 20px 16px 40px; }
  .topbar { align-items: stretch; flex-direction: column; }
  .run-picker { min-width: 0; width: 100%; }
  .timeline-meta { align-items: flex-start; flex-direction: column; gap: 4px; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .metric:nth-child(2) { border-right: 0; }
  .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
  .inspector-grid { grid-template-columns: 1fr; }
  .section-head { align-items: flex-start; flex-direction: column; }
  .badges { justify-content: flex-start; }
}
</style>
</head>
<body>
<main class="shell">
  <header class="topbar">
    <div>
      <h1>Cournot Event Playback</h1>
      <p class="subtitle" id="game-name">Repeated quantity market</p>
    </div>
    <div class="run-picker" id="run-picker">
      <label for="run-select">Experiment run</label>
      <select id="run-select"></select>
    </div>
  </header>

  <section class="timeline" aria-label="Playback timeline">
    <div class="timeline-meta">
      <span class="event-name" id="event-name">Event</span>
      <span class="event-count" id="event-count">0 / 0</span>
    </div>
    <input class="slider" id="timeline-slider" type="range" min="0" max="0" value="0" step="1" aria-label="Event position">
    <div class="controls">
      <button class="icon-button" id="previous-button" type="button" title="Previous event" aria-label="Previous event">&#9664;</button>
      <button class="icon-button primary" id="play-button" type="button" title="Play" aria-label="Play">&#9654;</button>
      <button class="icon-button" id="next-button" type="button" title="Next event" aria-label="Next event">&#9654;&#124;</button>
      <div class="speed">
        <label for="speed-select">Speed</label>
        <select id="speed-select">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
    </div>
  </section>

  <section class="section" aria-labelledby="market-heading">
    <div class="section-head">
      <h2 id="market-heading">Market State</h2>
      <div class="badges">
        <span class="badge" id="round-badge">Round 1</span>
        <span class="badge" id="treatment-badge">Treatment</span>
        <span class="badge provisional" id="stage-badge">Provisional</span>
      </div>
    </div>
    <div class="metrics">
      <div class="metric"><span class="metric-name">Total quantity</span><span class="metric-value" id="total-quantity">--</span></div>
      <div class="metric"><span class="metric-name">Price</span><span class="metric-value" id="price">--</span></div>
      <div class="metric"><span class="metric-name">Market profit</span><span class="metric-value" id="market-profit">--</span></div>
      <div class="metric"><span class="metric-name">Event type</span><span class="metric-value" id="event-type">--</span></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Player</th><th>Agent</th><th>Type</th><th class="number">Quantity</th><th class="number">Profit</th><th>Status</th></tr></thead>
        <tbody id="player-table"></tbody>
      </table>
    </div>
  </section>

  <section class="section" aria-labelledby="inspector-heading">
    <div class="section-head">
      <h2 id="inspector-heading">Decision Inspector</h2>
      <div class="badges" id="decision-badges"></div>
    </div>
    <div class="inspector-grid">
      <div>
        <div class="facts" id="event-facts"></div>
        <p class="reasoning" id="reasoning">Settlement event</p>
      </div>
      <div>
        <div class="data-block"><h3>Observation</h3><pre id="observation-data">--</pre></div>
        <div class="data-block"><h3>Actions and metadata</h3><pre id="action-data">--</pre></div>
      </div>
    </div>
  </section>
</main>

<script id="playback-data" type="application/json">__PLAYBACK_DATA__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("playback-data").textContent);
  const elements = {
    runSelect: document.getElementById("run-select"),
    runPicker: document.getElementById("run-picker"),
    gameName: document.getElementById("game-name"),
    slider: document.getElementById("timeline-slider"),
    previous: document.getElementById("previous-button"),
    play: document.getElementById("play-button"),
    next: document.getElementById("next-button"),
    speed: document.getElementById("speed-select"),
    eventName: document.getElementById("event-name"),
    eventCount: document.getElementById("event-count"),
    roundBadge: document.getElementById("round-badge"),
    treatmentBadge: document.getElementById("treatment-badge"),
    stageBadge: document.getElementById("stage-badge"),
    totalQuantity: document.getElementById("total-quantity"),
    price: document.getElementById("price"),
    marketProfit: document.getElementById("market-profit"),
    eventType: document.getElementById("event-type"),
    playerTable: document.getElementById("player-table"),
    decisionBadges: document.getElementById("decision-badges"),
    eventFacts: document.getElementById("event-facts"),
    reasoning: document.getElementById("reasoning"),
    observation: document.getElementById("observation-data"),
    action: document.getElementById("action-data")
  };
  let runIndex = 0;
  let eventIndex = 0;
  let timer = null;

  const currentRun = () => data.runs[runIndex];
  const currentStep = () => currentRun().timeline[eventIndex];
  const formatNumber = (value) => value === null || value === undefined ? "--" : Number(value).toFixed(2);
  const quantityFrom = (action) => action && action.value ? action.value.quantity : null;

  function addBadge(text, className = "") {
    const badge = document.createElement("span");
    badge.className = `badge ${className}`.trim();
    badge.textContent = text;
    elements.decisionBadges.appendChild(badge);
  }

  function addFact(label, value) {
    const labelElement = document.createElement("div");
    labelElement.className = "fact-label";
    labelElement.textContent = label;
    const valueElement = document.createElement("div");
    valueElement.className = "fact-value";
    valueElement.textContent = value;
    elements.eventFacts.append(labelElement, valueElement);
  }

  function addCell(row, text, className = "") {
    const cell = document.createElement("td");
    cell.className = className;
    cell.textContent = text;
    row.appendChild(cell);
  }

  function renderPlayers(step) {
    elements.playerTable.replaceChildren();
    step.players.forEach((player) => {
      const row = document.createElement("tr");
      if (player.active) row.className = "active";
      const identity = document.createElement("td");
      const playerName = document.createElement("div");
      playerName.className = "player";
      playerName.textContent = player.player_id;
      identity.appendChild(playerName);
      row.appendChild(identity);
      addCell(row, player.agent, "agent");
      addCell(row, player.agent_type);
      addCell(row, formatNumber(player.quantity), "number");
      const profit = step.market.stage === "settled"
        ? formatNumber(player.profit)
        : player.profit === null || player.profit === undefined
          ? "Pending"
          : `${formatNumber(player.profit)} previous`;
      addCell(row, profit, "number");
      addCell(row, player.state, `state ${player.state}`);
      elements.playerTable.appendChild(row);
    });
  }

  function renderInspector(step) {
    const event = step.event;
    elements.decisionBadges.replaceChildren();
    elements.eventFacts.replaceChildren();
    if (event.event_type === "decision") {
      addBadge(event.revision_allowed ? "Revision allowed" : "Forced hold", event.revision_allowed ? "" : "provisional");
      addBadge(event.valid === false ? "Invalid" : "Valid", event.valid === false ? "provisional" : "settled");
      addFact("Player", event.player_id || "--");
      addFact("Returned quantity", formatNumber(quantityFrom(event.returned_action)));
      addFact("Applied quantity", formatNumber(quantityFrom(event.applied_action)));
      addFact("Best reply", formatNumber(event.best_reply_quantity));
      addFact("Retry count", String(event.metadata.retry_count || 0));
      addFact("Latency", event.metadata.latency_seconds === undefined ? "--" : `${Number(event.metadata.latency_seconds).toFixed(3)} s`);
      elements.reasoning.textContent = event.reasoning || "No reasoning recorded.";
      elements.observation.textContent = JSON.stringify(event.observation, null, 2);
      elements.action.textContent = JSON.stringify({
        returned_action: event.returned_action,
        applied_action: event.applied_action,
        metadata: event.metadata,
        invalid_reason: event.invalid_reason
      }, null, 2);
    } else {
      addBadge("Round settled", "settled");
      addFact("Total quantity", formatNumber(event.total_quantity));
      addFact("Price", formatNumber(event.price));
      addFact("Market profit", formatNumber(event.total_market_profit));
      addFact("Distance to Nash", formatNumber(event.distance_to_nash));
      elements.reasoning.textContent = "All decisions for this round have been settled.";
      elements.observation.textContent = "No player observation for a settlement event.";
      elements.action.textContent = JSON.stringify(event, null, 2);
    }
  }

  function render() {
    const run = currentRun();
    const step = currentStep();
    const settled = step.market.stage === "settled";
    const last = step.market.last_settlement;
    elements.gameName.textContent = `${run.game_name} | ${run.run_id}`;
    elements.eventName.textContent = step.event_type === "decision"
      ? `${step.event.player_id} decision`
      : `Round ${step.round + 1} settlement`;
    elements.eventCount.textContent = `Event ${eventIndex + 1} / ${run.timeline.length}`;
    elements.slider.value = String(eventIndex);
    elements.roundBadge.textContent = `Round ${step.round + 1}`;
    elements.treatmentBadge.textContent = step.treatment ? `${step.treatment} treatment` : "Cournot";
    elements.stageBadge.textContent = settled ? "Settled" : "Provisional";
    elements.stageBadge.className = `badge ${settled ? "settled" : "provisional"}`;
    elements.totalQuantity.textContent = settled
      ? formatNumber(step.market.total_quantity)
      : `${formatNumber(step.market.provisional_total_quantity)} provisional`;
    elements.price.textContent = settled
      ? formatNumber(step.market.price)
      : last ? `${formatNumber(last.price)} previous` : "Pending";
    elements.marketProfit.textContent = settled
      ? formatNumber(step.market.total_market_profit)
      : last ? `${formatNumber(last.total_market_profit)} previous` : "Pending";
    elements.eventType.textContent = settled ? "Settlement" : "Decision";
    elements.previous.disabled = eventIndex === 0;
    elements.next.disabled = eventIndex === run.timeline.length - 1;
    renderPlayers(step);
    renderInspector(step);
  }

  function stopPlayback() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    elements.play.innerHTML = "&#9654;";
    elements.play.title = "Play";
    elements.play.setAttribute("aria-label", "Play");
  }

  function advance() {
    if (eventIndex >= currentRun().timeline.length - 1) {
      stopPlayback();
      return;
    }
    eventIndex += 1;
    render();
  }

  function startPlayback() {
    if (eventIndex >= currentRun().timeline.length - 1) eventIndex = 0;
    const speed = Number(elements.speed.value);
    timer = window.setInterval(advance, 900 / speed);
    elements.play.innerHTML = "&#10074;&#10074;";
    elements.play.title = "Pause";
    elements.play.setAttribute("aria-label", "Pause");
    render();
  }

  function selectRun(index) {
    stopPlayback();
    runIndex = index;
    eventIndex = 0;
    elements.slider.max = String(Math.max(0, currentRun().timeline.length - 1));
    render();
  }

  data.runs.forEach((run, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${run.game_name} | ${run.run_id}`;
    elements.runSelect.appendChild(option);
  });
  elements.runPicker.hidden = data.runs.length === 1;
  elements.runSelect.addEventListener("change", () => selectRun(Number(elements.runSelect.value)));
  elements.slider.addEventListener("input", () => {
    stopPlayback();
    eventIndex = Number(elements.slider.value);
    render();
  });
  elements.previous.addEventListener("click", () => {
    stopPlayback();
    eventIndex = Math.max(0, eventIndex - 1);
    render();
  });
  elements.next.addEventListener("click", () => {
    stopPlayback();
    eventIndex = Math.min(currentRun().timeline.length - 1, eventIndex + 1);
    render();
  });
  elements.play.addEventListener("click", () => timer === null ? startPlayback() : stopPlayback());
  elements.speed.addEventListener("change", () => {
    if (timer !== null) {
      stopPlayback();
      startPlayback();
    }
  });

  window.cournotPlayback = {
    setEvent(index) {
      stopPlayback();
      eventIndex = Math.max(0, Math.min(currentRun().timeline.length - 1, Number(index)));
      render();
    },
    getState() { return { runIndex, eventIndex, playing: timer !== null, step: currentStep() }; }
  };
  selectRun(0);
})();
</script>
</body>
</html>
"""

