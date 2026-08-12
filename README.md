## Simulating bounded rationality for LLM agents

The runner supports auction experiments and a Cournot oligopoly experiment based
on Huck, Normann, and Oechssler (1999). Both games use the same `Agent`,
`Participant`, `Environment`, serializer, and scorecard pipeline.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the component map and extension flow.

Run the Cournot experiment with local best-reply agents:

```bash
python3 simulation.py examples/cournot_run.json --analyze
```

Two additional local scripted policies are available:

```bash
python3 simulation.py examples/cournot_random_run.json --analyze
python3 simulation.py examples/cournot_previous_average_run.json --analyze
```

`cournot_random_quantity` samples a legal quantity between 1 and 100.
`cournot_previous_average` uses the preceding round's average quantity across
all four firms, falling back to its configured initial quantity in round one.

Run the resumable local Llama sweep with:

```bash
py scripts/run_cournot_sweep.py examples/cournot_llama_sweep.json --dry-run
py scripts/run_cournot_sweep.py examples/cournot_llama_sweep.json
```

The default study runs `llama3.1:8b` against homogeneous and mixed scripted
populations over four temperatures, one-to-four model firms, BEST/FULL
treatments, and five trials. It checkpoints every run beneath
`runs/cournot-sweeps/`. Resume an interrupted sweep using the directory printed
by the runner:

```bash
py scripts/run_cournot_sweep.py --resume runs/cournot-sweeps/SWEEP_DIRECTORY
```

Use `--limit N` for a smaller smoke run and `--workers N` to override the
default two concurrent runs. Completed sweeps contain raw run bundles plus
player, run, and treatment-cell CSV summaries and a self-contained Plotly
dashboard under `analysis/`.

Cournot model agents receive all completed rounds by default. In JSON, set
`memory_rounds` to `null` for the full history, `0` for no completed-round
history, or a positive integer for only the most recent N rounds. The browser's
memory field follows the same rule: leave it blank to include all rounds. The
local Ollama profile and sweep use a 16,384-token context to accommodate the
growing 40-round history.

Or use four OpenAI-compatible local models:

```bash
python3 simulation.py examples/cournot_llm_run.json
```

Run one human against three OpenAI-compatible local models with:

```bash
python3 simulation.py examples/cournot_human_vs_llm_run.json --analyze
```

Each run writes `participant_data.csv`, `system_data.csv`, `events.jsonl`, and
`summary.json`. Cournot runs automatically write `analysis/scorecard.json`,
`analysis/scorecard.csv`, `analysis/cournot_player_scores.csv`,
`analysis/cournot_player_scores.svg`, and
`analysis/cournot_player_scores_ci95.svg`. The frontend can switch the graph
whiskers between two standard deviations and a 95% Student-t confidence
interval. Cournot analyses also include a self-contained
`analysis/cournot_playback.html` for replaying every decision and settlement.
The graph shows average profit per player and lets the frontend switch between
two-standard-deviation whiskers and 95% confidence intervals.
Choose another analysis directory with:

```bash
python3 simulation.py examples/cournot_run.json --analysis-output analysis_output/cournot
```

Analyze or compare existing run directories independently with:

```bash
python3 -m sandbox.scorecard RUN_DIR [RUN_DIR ...] -o analysis_output/comparison
```

View completed Cournot graphs and event playbacks in the frontend with:

```bash
python3 -m pip install -r frontend/requirements.txt
python3 -m uvicorn webapp:app --reload
```

Open `http://127.0.0.1:8000`, switch the experiment control to `Cournot`,
configure hyperparameters, then assign each player as a human, model, or script.
The Agent Assignment tab can add or remove firms from the supported two-to-eight
player range; four remains the paper's default. The separate Agent Prompts tab
shows prefilled system and decision prompt editors for every active player, so
model players can use different instructions without hidden override controls.
Each model player also has the same provider-profile dropdown as Auction. It
supports configured local Ollama/OpenAI-compatible, OpenAI, Anthropic/Claude,
and Gemini profiles from `config/providers.json`; unavailable profiles remain
visible but disabled until their model and API-key environment variables exist.
Browser-human turns appear on the page while the job is running. Dashboard jobs
use macOS `caffeinate` to keep the computer awake until execution finishes. The
round progress bar advances after each settlement, and the graph and playback
appear only after that job completes.

### Contributors

- [Aryan Singh](https://www.linkedin.com/in/aryanmsingh/)
- [Dhruva Bhat](https://www.linkedin.com/in/dhruvabhat/)
- [Dr. Sarah H. Cen](https://www.linkedin.com/in/sarah-cen-70394869/)
