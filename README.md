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
`summary.json`. With `--analyze`, the run also writes `analysis/scorecard.json`
and `analysis/scorecard.csv`. Choose another analysis directory with:

```bash
python3 simulation.py examples/cournot_run.json --analysis-output analysis_output/cournot
```

Analyze or compare existing run directories independently with:

```bash
python3 -m sandbox.scorecard RUN_DIR [RUN_DIR ...] -o analysis_output/comparison
```

### Contributors

- [Aryan Singh](https://www.linkedin.com/in/aryanmsingh/)
- [Dhruva Bhat](https://www.linkedin.com/in/dhruvabhat/)
- [Dr. Sarah H. Cen](https://www.linkedin.com/in/sarah-cen-70394869/)
