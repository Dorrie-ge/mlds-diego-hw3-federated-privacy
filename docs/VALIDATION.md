# Validation experiments and reproduction

Every result below is retained as measured JSON/JSONL. These tests have specific
algorithmic predictions; they are not merely file-format or installation tests.

| Control | Prediction and interpretation | Command / evidence |
|---|---|---|
| Unequal sample weighting | Weights 2,10 with counts 1,3 must aggregate to 8, not the unweighted 6 | `python validate.py --ray`; `results/validation/algorithm_validation.json` |
| Gradient equivalence | One full-batch local SGD step from the same initialization on unequal-size synthetic clients equals pooled SGD; not generally expected after multiple local steps | Same command; `full_batch_gradient_identity` |
| No updates | E=0 and learning rate=0 preserve parameters; the real-data zero-rate trajectory must keep validation metrics fixed | `validate.py`; `validate_course.py`; `zero_lr/` |
| Known learnable task | A separable two-class one-hot fixture should exceed .95 accuracy and lower validation loss in ten rounds through the actual FedAvg path | `validate.py`; explicitly synthetic results |
| Laplace endpoints/moments | b=0 preserves inputs exactly; 100,000 Laplace(0,.2) draws should have mean 0 and variance .08 | `validate.py`; `laplace_moments` |
| Real b=0 identity | Zero-noise and matching clean FedAvg checkpoints and all round metrics agree exactly under every seed | `python validate_crosschecks.py`; `course_crosschecks.json`; local ZIP checkpoints or reproduced training |
| Serial/Ray parity | Matched data, initial state, selected writers, minibatches and seeds produce identical parameters/metric trajectories | `analysis.py`; `report_metrics.json`; `ray_parity.png` |
| Real concurrency | Actor start/finish intervals actually overlap; record observed concurrency rather than assume it from API names | Client JSONL timestamps and `report_metrics.json` |
| Label permutation | Shuffling labels inside each writer destroys image-label association but preserves local label histograms; clean validation should fall far below normal learning, not necessarily to 1/62 | `python validate_course.py --data data/prepared`; `shuffled_labels/` |
| Extended local updates | E=10 versus E=3 at 40 rounds holds sampling fixed but changes compute; inspect norms and trajectories without requiring a harmful effect | `large_E10/` versus matched E=3 seed 17 |
| Participation/heterogeneity | Uniform sampling in expectation need not give uniform finite-run coverage; measure actual counts and writer/global label total-variation distance | `analysis.py`; `heterogeneity_coverage.png` |
| Noise boundaries | Added pixels can leave [0,1]; measure fraction and absolute noise and separate display clipping from unclipped training | `analysis.py`; `noise_samples.png` |

## First ten rounds and non-standard plots

Read `history.json` rounds 0-10 and `client_updates.jsonl` rounds 1-10.
Each selected client record includes sample count, optimizer-step count, sample
weight, update norm, local/global cosine, and actual start/finish time. The custom
alignment plot shows cosine versus round, colored by norm and sized by sample
weight. It is generated for serial, Ray and noisy models. A cosine of 0 for a
zero-length update is a convention, not a meaningful direction. Cross-panel
colors are normalized separately; compare JSON values for numerical norm claims.

## Separation, edge cases and limitations

Writer-specific 80/20 splits use a fixed seed and disjoint/exhaustive indices.
The official test partition is read only by final evaluation after validation
selection. Hash/schema inspection is not model tuning. A deliberately invalid
test archive in the synthetic fixture confirms that the training path does not
read test data. This is a local simulation, not security against a malicious host.

The loader rejects tiny clients (fewer than two images), invalid labels,
nonfinite/out-of-range clean pixels and an incorrect user count. Missing local
classes are valid, and the final short minibatch is retained. Selection is without
replacement, optimizer state resets per local update, and actors exist only for
selected users. Noise is unclipped; b=0 is exact; negative/nonfinite b is rejected.

Three seeds and one partition are finite evidence. The full Ray run has one
seed. Synthetic exact identities do not prove EMNIST generalization. Higher E
changes compute. Positive alignment may reflect a shared wrong prior. Labels
and clean metric releases are not privatized, and pixel noise alone does not
establish user-level DP. Public reproducibility seeds are not secret privacy
randomness. The report proposes additional splits/seeds, equal-budget studies,
missing-client stress, calibrated privacy accounting and fresh test data.

## Complete auxiliary reproduction and diagnostic summaries

Run `python validate_adapters.py` for all synthetic JSON/HDF5/NPY, exact indexing,
8/2 split and deterministic repetition checks. Run `python validate_crosschecks.py`
after training (or from the local ZIP with weights) to reproduce the three
b=0/clean parameter and 121-point history comparisons. Both write their full
results under `results/validation/`; their execution logs are in `docs/execution/`.
The adapter tests are supplementary data checks, not a replacement for the
algorithmic identities and real-label counterfactual in this guide.

Run `python diagnostics.py` for the exact numbers underlying the report's new
quantitative page. It needs only saved JSONL and exports JSON plus CSV for every
retained run. It fixes windows to the first/last ten rounds and, where available,
the first 40 rounds for the E stress comparison. Formula definitions and limits
are stored with the JSON. It neither selects models nor opens test images.

The sample-weighted mean cosine is a mean across rounds of the weighted client
cosines; the negative-update fraction counts selected updates equally. The output
also retains sample-weighted negative shares, p90 norms and zero-norm counts.
Ray/serial matched four-client summaries should agree exactly. Higher cosine
across different client counts is not a systems or accuracy improvement, and a
larger norm with higher E is not in itself evidence of harmful drift.
