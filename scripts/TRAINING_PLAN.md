# Training plan — bloom_256 encodings

## State of play

| Location | What's there | Reusable? |
|---|---|---|
| `boosting_runs/classification_sigma/0.5_sigma/` | 15 runs: Bloom2013, Bloom2015, Bloom2019_BYxRM × 5 seeds | **No** — trained on the superseded data |
| `boosting_runs/classification_encodings/` | two empty `sigma_*` dirs | nothing run yet |
| `boosting_runs/classification_other_models/` | empty | — |

The 15 existing runs cannot serve as the `max` baseline — see the difference table below.
Keep them as the record of the old results; every number in the new comparison has to come from
a fresh run.

**Scope: Bloom2013, Bloom2015, Bloom2019_BYxRM only.** BYxM22 and RMxYPS163 are commented out of
all three run scripts (the same way `run_script.sh` does it), so re-enabling them later is a matter
of uncommenting two blocks. Their converted data is already on disk either way.

**So: 18 run-groups to do** (3 datasets × 3 encodings × 2 thresholds), each 5 seeds = 90 runs.

## How different is the new data from what you trained on?

Compared block by block against `full/varying_sigma/sigma_0.5/*_clf.feather`, keyed naturally
(genotype per strain, chemistry per condition) rather than by row position:

| | Bloom2013 | Bloom2015 | Bloom2019_BYxRM |
|---|---|---|---|
| rows | 23660 (=) | 86938 (**+46**) | 18114 (=) |
| strains / conditions | identical | identical | identical |
| **genotype cells changed** | 0.118% | **0.000%** | **3.503%** |
| genes affected | 14 / 6014 | 0 / 6014 | **444 / 6014** |
| strains affected | all 1007 | none | all 951 |
| direction | all upward | — | all upward |
| **chemistry, mean abs diff** | 11.9% of mean value | 11.6% | 11.2% |
| conditions with identical latent | 2 / 39 | 1 / 18 | 0 / 31 |
| phenotype labels | identical | 52 of 86515 aligned rows differ | identical |

Reading it:

- **Chemistry moved for every dataset, ~12% in relative terms.** This is the regenerated
  embedding, not noise — `latent_pubchem_256.csv` is stamped two minutes *after* the training
  feathers were written, and the version that built them no longer exists. This alone means no
  `max` result is reproducible against the old numbers.
- **Bloom2015's genotype is bit-identical** — 0 of 26M cells changed. Its only movement is 46 extra
  rows and 52 flipped labels out of 86515, i.e. a phenotype/replicate fix, nothing structural.
- **Bloom2019_BYxRM changed most, by a wide margin**: 3.5% of gene cells across 444 genes, touching
  every strain. The dominant transition is `0 → 1` (129277 cells), i.e. genes the old build scored
  as unmutated that the fresh snpEff annotation now scores. Expect this cross's results to move
  most, and say so in the write-up.
- **Bloom2013's 14 changed genes are all `→ 3`**, the `effects.xlsx` duplicate-reduction fix.
- Nothing was ever removed: every genotype change in both affected datasets is upward.

And the reviewer's actual question — do the new encodings carry information `max` throws away?

| | sum vs max | genes affected | largest gene burden |
|---|---|---|---|
| Bloom2013 | 9.39% of cells differ | 1125 / 6014 | 12 (max encoding caps at 3) |
| Bloom2015 | 13.99% | 1682 / 6014 | 19 |
| Bloom2019_BYxRM | 17.47% | 2085 / 6014 | 36 |

So yes — between 9% and 17% of gene cells carry burden that `max` flattens, and `count` is strictly
richer than `sum`. The comparison is worth running.

## Order

1. **sigma_0.5 first** — that's the near-balanced split the current work uses. All three encodings.
2. **sigma_1.0 second** — same 9 groups at the other threshold. Roughly half the rows, so about
   half the time. Only needed if you want the threshold comparison in the paper.

Within a threshold, run `max` first — it is the baseline the reviewer's two new encodings get
read against, so if you stop early you still have something to compare to.

## How to run it

Running inside zellij, so no `nohup` — the pane keeps the process alive and you keep the
scrollback. One encoding at a time, waiting for each to finish:

```bash
cd /home/rajeeva/projects/Boosting-yeast_growth_pred
pixi shell
mkdir -p logs

./scripts/run_encodings_max.sh   0.5 2>&1 | tee logs/max_0.5.log
./scripts/run_encodings_sum.sh   0.5 2>&1 | tee logs/sum_0.5.log
./scripts/run_encodings_count.sh 0.5 2>&1 | tee logs/count_0.5.log
```

Or hand the whole threshold to one command — `run_encodings_all.sh` already runs the three in
this order, sequentially:

```bash
./scripts/run_encodings_all.sh 0.5 2>&1 | tee logs/all_0.5.log
```

Then repeat with `1.0`. `tee` keeps the output on screen and on disk, so `grep Done logs/*.log`
works afterwards even if the scrollback is gone — each script echoes one line per finished dataset.

One stream uses `NUM_THREADS=24` of your 112 cores and peaks around 45 GB (count on Bloom2015),
so there is plenty of headroom if you later decide to put a second encoding in another pane.

## Time and resources

Calibrated on Bloom2013 at sigma_1.0 (11 959 rows), 2 Optuna trials, 1 seed:

| encoding | wall | peak RSS |
|---|---|---|
| max (6 273 cols) | 49 s | 2.7 GB |
| count (18 301 cols) | 67 s | 6.5 GB |

Extrapolating to 100 trials × 5 seeds and scaling by row count (sigma_0.5 totals 128 712 rows
across the three datasets — Bloom2015 alone is two thirds of it), running one stream at a time:

| threshold | max | sum | count | total |
|---|---|---|---|---|
| **sigma_0.5** | ~22 h | ~22 h | ~33 h | **~77 h (3+ days)** |
| sigma_1.0 | ~11 h | ~11 h | ~16 h | ~38 h |

Treat these as order-of-magnitude: two trials is a poor sample when Optuna is sampling
`n_estimators` from 10–1000 and `num_leaves` from 2–256. The first `Done` line in each log gives
you a real per-dataset number — re-estimate then, before committing to sigma_1.0.

**Sequentially that is ~5 days for both thresholds.** If that is too much, in order of preference:
- skip sigma_1.0 entirely — sigma_0.5 answers the reviewer's question on its own, and this is the
  single biggest saving
- `n_trials=50` appended to the count runs only — halves the most expensive stream
- drop to 3 seeds (`'seed=1,2,3'`) — costs you the error bars
- put `sum` in a second zellij pane while `max` runs; they are independent and the box has room

## Restart / gap-filling

`tune_model.py` overwrites, and there is no resume. If a stream dies, re-run just what's missing —
each dataset is one command:

```bash
python src/tune_model.py --config-name=conf_encodings --multirun \
  dataset=bloom2015_count alpha=0.5 'seed=1,2,3,4,5'
```

A group is complete when its directory holds `Boosting.pkl`, `Boosting_best_params.pkl` and
`Boosting_run_report.html`. To find gaps:

```bash
for d in $RUN_DIR/classification_encodings/sigma_0.5/*/; do
  [ -f "$d/Boosting.pkl" ] || echo "INCOMPLETE $d"
done
ls $RUN_DIR/classification_encodings/sigma_0.5 | wc -l   # expect 45 (9 groups x 5 seeds)
```

## After training — evaluation

`compare_performance.py` reads `configs/eval.yaml`, which pulls in `utilities.yaml` and the
**regression** metric set. It needs a companion config pointing at the new runs with `metrics: clf`
— one small new file, `configs/eval_encodings.yaml`, along the lines of:

```yaml
defaults:
  - eval
  - utilities_encodings
  - _self_
  - metrics: clf
```

then supply `data_paths` and `model_load_keys.model_names` on the command line the way
`scripts/evaluate.sh` does. `model_load_keys.prefix` already resolves to `${encoding_prefix}_`
(`Max_`, `Sum_`, `Count_`), so it will find the right run directories per encoding.

Ask me to write that config and an `evaluate_encodings.sh` when the runs are underway — it is
quick, and it is easier to get right once there are real directories to glob against.

## Reporting caveat to carry into the write-up

The `max` numbers are a **new baseline**, not a reproduction of the published ones. State plainly
that the chemical embedding moved (~12% in relative terms, every dataset) and that the genotype
matrices changed for two of the three — quote the BYxRM figure (3.50% of gene cells, 444 genes,
every strain), since that cross moved most and its results will shift most. Bloom2015's genotype
is unchanged, which is worth saying too: it isolates the embedding effect cleanly.
