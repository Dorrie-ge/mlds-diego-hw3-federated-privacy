# Assignment 3 - Federated Learning and Differential Privacy

GitHub username: **Dorrie-ge**. Repository: **mlds-diego-hw3-federated-privacy**.

**[Read the complete PDF report](report.pdf)** · [Validation guide](docs/VALIDATION.md)

I implement FedAvg directly in PyTorch, run four Ray Core client actors per
round, and compare fixed Laplace input perturbations at five scales on the exact
100-writer course Federated EMNIST dataset. This is Diego HW3, following
`mlds_assignment3.pdf`. The report contains measured results, required figures,
custom diagnostics, edge cases, quantitative validation, and my three-month plan.

## Data and protocol

The Canvas archive contains 31,825 local images across 100 writers and 3,621
independent held-out test images. My fixed writer-specific 80/20 splits produce
25,418 training and 6,407 validation examples. Pixels are float32 in [0,1]; I
flatten them without changing orientation or polarity. Labels remain 0-61.
See `docs/DATA_PROVENANCE.json` for archive/raw-file hashes and provenance.
I exclude course raw data and checkpoint binaries from GitHub.

The model is 784 -> 128 -> ReLU -> 62 (two affine layers, 108,478 parameters).
Local SGD uses learning rate .05, batch size 32 and a fresh optimizer each round.
I aggregate parameters by actual local training sample counts. Clients return
states, counts and scalar evaluation summaries, not images to another client.
All-client metrics evaluate the same current global model with sample weights.
This is a local simulation, not a secure multiparty deployment.

I compare C={.04,.10} and E={1,3} over 120 rounds with seeds 17,29,43, choosing
by final mean validation accuracy. Noise runs use C=.10, the selected E, and
b={0,.05,.10,.30,1.0}. Each image receives one fixed independent perturbation per
run, reused across rounds, with no clipping of actual training values. Validation
and test inputs remain clean. Histories distinguish noisy `train_accuracy` from
`train_clean_accuracy`. My utility rule chooses the largest nonzero b within two
validation accuracy points of b=0, if any. Test accuracy never selects settings.

The full Ray experiment uses seed 17, four clients per round, the selected E,
and 120 rounds. I instantiate actors only for selected clients, reserve one CPU
and zero GPUs per actor, wait for readiness and kill actors after each round.
The matched serial/Ray check uses identical four-client schedules and seeds.
It does not compare four-client Ray to a ten-client serial run for equality.

## Reproduce

Use Python 3.12 and a clean environment. `requirements-lock.txt` lists actual
training/plot versions; `requirements.txt` gives compatible ranges.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python prepare_data.py --train /path/Assignment3-data/train_data.npy --test /path/Assignment3-data/test_data.npy --pixel-divisor 1 --trusted-course-npy --output data/prepared
python validate.py --ray
python validate_adapters.py
python validate_edges.py
python validate_course.py --data data/prepared
python experiments.py train --data data/prepared --out results/course --rounds 120
python validate_crosschecks.py
python diagnostics.py
python experiments.py test --data data/prepared --out results/course
python plots.py --data data/prepared --results results/course
python analysis.py --data data/prepared --results results
python report.py --repo-url https://github.com/Dorrie-ge/mlds-diego-hw3-federated-privacy --font-dir /path/to/TimesNewRomanFonts
```

Only load object NPY files from the verified Canvas archive with
`--trusted-course-npy`: their NumPy object arrays require pickle loading. The
adapter also supports LEAF JSON and TFF HDF5. Do not substitute ordinary EMNIST
or silently subsample a different writer collection.

Use fresh output directories when rerunning, or archive original results first.
Completed runs are reused only when configs and dataset manifests match.
Move interrupted folders aside before restarting. Test evaluation refuses to
overwrite an existing result. Frozen test-model choices are in `selection.json`,
whose hash is recorded in `heldout_test.json`.

PDF regeneration requires licensed Times New Roman regular/bold/italic TTFs;
macOS's standard font directory is the default. Font binaries are not distributed.
Plots use installed Times New Roman on the original machine, otherwise a default
font. No retraining is needed to rebuild the PDF from retained figures/JSON.

## Files and evidence

| Location | What I retain |
|---|---|
| `federated.py`, `prepare_data.py` | Model, local training, aggregation, Ray, splits and data adapters |
| `experiments.py` | Validation-only sweeps and separate frozen test evaluation |
| `validate.py`, `validate_course.py` | Synthetic prediction controls and real-data counterfactuals |
| `validate_adapters.py`, `validate_crosschecks.py` | Complete reproduction of adapter/index checks and b=0 checkpoint/history equality |
| `diagnostics.py` | Numerical analysis of early/late alignment, negative shares and update norms |
| `plots.py`, `analysis.py`, `report.py` | Reproducible required/custom figures, tables and report |
| `results/course/` | Configs, all round histories, client traces, participation, test results |
| `results/course_controls/` | Label permutation, high-E and zero-rate experiments |
| `results/validation/` | Executed arithmetic, gradient, noise, learning and parallel controls |
| `results/report_metrics.json` | Computed report values and serial/Ray equality |
| `results/course/figures/` | Required plots and non-standard diagnostics |
| `docs/` | Data provenance, validation directions and execution logs |

I label synthetic results explicitly and never report them as EMNIST accuracy.
The weighted-average oracle returns 8 rather than 6; the full-batch gradient
identity has maximum error 7.45e-9; zero-update errors are 0; Laplace variance
at b=.2 is .080426 versus .08 predicted; four synthetic Ray actors match serial
updates exactly. The real-data label permutation preserves writer class priors
but destroys sample correspondence. Higher E is a communication-budget stress,
not an equal-compute performance comparison. See the report for actual outcomes.

My custom alignment plot shows each client's cosine with the global update,
colored by norm and sized by sample weight, for serial, Ray and noisy runs.
Other diagnostics measure writer heterogeneity, participation and noisy pixels
outside the clean domain. JSONL records include the first ten rounds.

## Auxiliary validations and quantitative diagnostic analysis

`python validate_adapters.py` creates its own temporary synthetic fixtures: 100
clients, ten samples each, independent held-out fixture, LEAF JSON, TFF HDF5 and
trusted object NPY formats. It checks exact arrays, labels and indices, disjoint
and exhaustive 8/2 splits, same-seed repetition and explicit pickle trust. It
writes `results/validation/adapter_validation.json`; no course data are needed.

`python validate_crosschecks.py` compares all three C=.10/E=3 clean and b=0
checkpoint pairs and all 121 saved metric points per pair. Counts, configs,
dataset manifests, finite tensor values and exact parameter/history equality are
checked. All errors are zero. Checkpoints are excluded from GitHub. After a fresh clone, obtain the course
data from Canvas and first run the documented training command. No separate
checkpoint download is required or assumed. Neither auxiliary script reads course
test images. Execution outputs are retained in `docs/execution/`.

`python diagnostics.py` recomputes `results/nonstandard_diagnostics.json` and
`.csv` from existing update logs, without checkpoints, course images or test
access. It covers every retained run. The report adds a dedicated quantitative
page: early rounds 1-10, late rounds 111-120, and the E=3/E=10 comparison at the
same first-40-round horizon. Sample-weighted cosine is averaged across rounds;
negative share is an unweighted fraction of selected updates; median norms are
unscaled local parameter L2 norms. These are descriptive post-training summaries,
not new model-selection criteria or causal tests. Comparing Ray C=.04 cosine
with C=.10 directly is confounded by each client's self-contribution to its own
aggregate; the serial/Ray equality check instead uses matched four-client runs.

## Privacy and experimental limits

I implement pixel perturbation, not a claim of image-label or user-level DP.
Labels and clean evaluation summaries are unprotected. With fixed public labels,
the elementary bounded-image Laplace sensitivity bound is epsilon=784/b, which
does not imply strong privacy at the utility-preserving scales. Public seeds
are for experiments, not secret deployment randomness. The b choice is an
explicit utility rule. Three seeds, one split and a finite round budget do not
establish convergence or state-of-the-art performance; full Ray uses one seed.

## References

- [McMahan et al., FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html)
- [Ray Core actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Dwork and Roth, differential privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- Caldas et al., LEAF: A Benchmark for Federated Settings, arXiv:1812.01097 (2018)

## Boundary-case evidence added November 18, 2026

Run `python validate_edges.py` without course data or checkpoints. Appendix A of
the revised PDF lists ten executed boundary controls with the test method,
expected behavior and measured result. Full machine-readable observations and
source hashes are in `results/validation/edge_validation.json`; the execution
log is `docs/execution/edge_validation.log`. The cases cover missing classes,
short minibatches, two-sample and one-sample clients, invalid labels/pixels,
participation limits, invalid aggregation counts, and noise boundaries.

This revision reruns the adapter checks and the three retained clean/b=0
checkpoint comparisons. It does not rerun the complete course training sweep.
Original training and evaluation records remain dated November 14, 2026.
