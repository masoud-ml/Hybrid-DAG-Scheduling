# Hybrid DAG Scheduling

A research implementation of fixed-priority, nonpreemptive scheduling for directed acyclic graph (DAG) tasks on homogeneous multicore systems. It provides an event-driven scheduler, a sufficient static timing-anomaly test, scenario generation and labeling, semi-supervised classification, and response-time bound evaluation.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`
- A C compiler may be needed on platforms without prebuilt wheels for scientific Python packages

The commands below assume the repository root as the working directory.

## Installation

Create and activate an isolated Python environment, then install the runtime dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick run

Generate and label a small dataset, then evaluate its response-time bounds:

```bash
python -m pipeline.reproduce --mode quick --out out/quick
```

The quick configuration generates 24 scenarios and uses at most 64 sampled executions for each scenario that the static test cannot resolve. To parallelize labeling:

```bash
python -m pipeline.reproduce --mode quick --out out/quick --workers 2
```

The pipeline writes generated DAGs, an index, labels and features, protocol metadata, and evaluation tables beneath `out/quick/`. Training is attempted when the labeled data contain enough examples from both classes; otherwise, the pipeline records why it was skipped. This is expected for a small sample and does not prevent labeling or evaluation.

The quick configuration is a smoke run. It does not reproduce the full experimental results or populate every fixed-parameter comparison slice.

## Analyze a DAG

Run the hybrid analysis on an included DAG:

```bash
python -m pipeline.hybrid_approach \
  --dag examples/demo_run/dags/sc0000001.json \
  --cores 2
```

The command prints a JSON result with the static-test outcome, WCET makespan, analytical bound, and selected bound. To select the generic list-scheduling bound, add `--analytic generic`.

DAG inputs are JSON files with nodes and directed edges. For example:

```json
{
  "dag_id": "chain",
  "nodes": [
    {"id": "source", "wcet": 1.0, "priority": 2},
    {"id": "sink", "wcet": 2.0, "priority": 1}
  ],
  "edges": [{"u": "source", "v": "sink"}]
}
```

WCET values must use consistent time units. Higher numeric values indicate higher priorities. Certified static analysis requires an acyclic graph with one source, one sink, and distinct priorities. Zero-cost boundary nodes must be marked with `"dummy": true`.

## Dataset workflows

Generate a small custom dataset, label it in parallel, and evaluate the results:

```bash
python -m pipeline.dataset_generation \
  --out out/custom \
  --length-min 4 --length-max 5 \
  --parallel-min 3 --parallel-max 4 \
  --core-min 2 --core-max 3 \
  --per-setting 10

python -m pipeline.dataset_labeling_mp \
  --index out/custom/index.csv \
  --out out/custom/labels.csv \
  --trials 1000 --workers 2

python -m evaluation.evaluate \
  --labels out/custom/labels.csv \
  --out out/custom/evaluation
```

For single-process labeling, use `pipeline.dataset_labeling_sp` with the same options except `--workers`. Each command provides `--help` for its arguments.

Labels mean:

- `0`: static analysis certifies anomaly freedom under the supported assumptions.
- `1`: a sampled execution exceeds the WCET-simulation makespan.
- `undefined`: neither condition has been established.

A failed static test does not prove an anomaly. Likewise, failing to observe an anomaly in sampled executions does not prove anomaly freedom. Positive labels include a replayable `witness_json`.

## Train and apply the classifier

After collecting sufficient labeled examples from both classes:

```bash
python -m pipeline.train_model \
  --labels out/custom/labels.csv \
  --out out/custom/learning

python -m pipeline.hybrid_approach \
  --dag examples/demo_run/dags/sc0000001.json \
  --cores 2 \
  --model out/custom/learning/trained_model.joblib
```

Training balances the labeled classes, reserves a validation partition, and uses high-confidence predictions from an undefined pool for self-training. A separate subset is held out for predictions on unseen scenarios. The output directory contains the model, split manifest, training history, classification metrics, feature importance, and available plots.

Reduced datasets may not meet the default sample sizes. Use `--strict-counts` when exact requested training-pool counts are required. Learning curves for rounds that include pseudo-labeled data are diagnostic; they are not independent accuracy estimates against ground-truth labels.

Classifier predictions are advisory. The pipeline uses a simulated makespan as a safe selected bound only when static analysis certifies the scenario; otherwise, it retains the analytical bound.

## Full experiment

The full parameter sweep generates 2,160,000 scenarios. Each statically unresolved scenario may require up to 100,000 sampled executions. Run the phases separately:

```bash
python -m pipeline.reproduce --mode paper --out out/full --phase generate
python -m pipeline.reproduce --mode paper --out out/full --phase label --workers 4
python -m pipeline.reproduce --mode paper --out out/full --phase train
python -m pipeline.reproduce --mode paper --out out/full --phase evaluate
```

Training in this mode requires 40,000 labeled scenarios per class, 80,000 undefined training scenarios, and 20,000 separate unseen scenarios. The full sweep is computationally and storage intensive. Options such as `--max-scenarios`, `--per-setting`, and `--trials` reduce the experiment and should be reported when presenting results. `--max-scenarios` limits new records per invocation, not the cumulative dataset size.

## Outputs and resuming

Generated data and results are written under the output directory supplied with `--out`. The orchestrator resumes generation and labeling when configuration and inputs match. Keep the generated manifests and protocol files with the outputs. If the code, parameters, or input index changes, use a new output directory rather than combining incompatible results.

## Reference bounds and limitations

- The priority-aware isolated-DAG bound follows Chen et al., [RTAS 2023](https://eprints.whiterose.ac.uk/id/eprint/201373/).
- The isolated-DAG specialization of the Serrano et al. [DATE 2016](https://hipert.unimore.it/people/marko/papers/DATE16.pdf) bound is `L + (W - L) / m`.
- The limited-preemption adaptation of He et al., [ECRTS 2021](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2021.8), is not included. Independently validated values can be provided through `--other-bounds-csv`; the required columns are `dag_id,bound_28`, with optional `bound_4`.

The reference paper does not specify all generation seeds, the exact WCET distribution, or every implementation choice. This repository makes explicit choices for those details. It does not claim exact reproduction of the paper’s reported accuracy or bound-reduction results.

## Tests

Install development dependencies and run the test suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover scheduler examples, graph metrics, static analysis, anomaly witnesses, response-time bounds, training data separation, and sequential/parallel labeling.

## Repository structure

| Path | Description |
| --- | --- |
| `core/` | DAG model |
| `schedule/` | Runtime and WCET schedulers |
| `analysis/` | Graph metrics, static analysis, and response-time bounds |
| `data/` | Scenario generation, features, and labeling |
| `learning/` | Random Forest classifier |
| `pipeline/` | Command-line workflows |
| `evaluation/` | Bound comparisons and plots |
| `utils/` | DAG I/O and supporting utilities |
| `examples/` | DAG input fixtures |
| `test/` | Automated tests |
