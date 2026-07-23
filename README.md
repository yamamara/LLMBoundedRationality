## Simulating bounded rationality for LLM agents

The runner supports auction experiments and a Cournot oligopoly experiment based
on Huck, Normann, and Oechssler (1999). Both games use the same `Agent`,
`Participant`, `Environment`, serializer, and scorecard pipeline.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the component map and extension flow.

Run the Cournot experiment with local best-reply agents:

```bash
python3 simulation.py examples/cournot_run.json --analyze
```

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
configure hyperparameters and editable prompts, then assign each player as a
human, model, or script. Browser-human turns appear on the page while the job
is running. The graph and playback appear only after that job completes.

### Contributors

- [Aryan Singh](https://www.linkedin.com/in/aryanmsingh/)
- [Dhruva Bhat](https://www.linkedin.com/in/dhruvabhat/)
- [Dr. Sarah H. Cen](https://www.linkedin.com/in/sarah-cen-70394869/)
