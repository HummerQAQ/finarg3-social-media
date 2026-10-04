# BAIBAICHUCHU at NTCIR-19 FinArg-3 — Social Media

Team **BAIBAICHUCHU** participated in the Social Media subtask of **FinArg-3,
a core task at NTCIR-19**. Our submission **ranked 1st in both real-time
evaluation settings**, and our participant paper was **selected for an oral
presentation**.

This repository accompanies the unpublished manuscript **“When Is Maximum
Possible Profit Predictable from Investor Text?”** by Zong-Han Bai and Po-Yen
Chu. It contains our three-track system—lexicon-based logistic regression,
a pre-finetuned MacBERT ranker, and an LLM judge—plus post-hoc analyses of
stance handling and volatility–horizon capacity.

**Task data is not included.** Obtain the restricted datasets from the
NTCIR-19 organizers as a registered participant.

## Setup

Use Python 3.11+ and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=...                      # only for LLM steps
```

Auxiliary pre-finetuning requires CUDA. Keep API keys in the environment;
organizer data, caches, checkpoints, outputs, and `.env` files remain untracked.

### Inputs

Set input locations explicitly; scripts do not guess alternate locations:

```bash
export FINARG3_RAW_DIR="/path/to/organizer-release"
export FINARG_EXTERNAL_DATA_DIR="/path/to/external_data"
export FINARG3_ORIGINAL_TEST="$FINARG3_RAW_DIR/Social_Media_Pairwise_Test_with_translation.json"
```

| Variable | Required contents / purpose |
|---|---|
| `FINARG3_RAW_DIR` | `Social_Media_Posts_with_MPP_ML.json` and `Social_Media_Pairwise_Test_with_translation.json` for data preparation |
| `FINARG_EXTERNAL_DATA_DIR` | `FinArg-2/Social Media/` with `IDED_Train.json`, `IDED_Dev.json`, and `IDED_Test_ANS.json`; relevance ablation also uses `FinArg-1/Social Media/train.json` and `dev.json` |
| `FINARG3_ORIGINAL_TEST` | Original organizer test file, whose fields are preserved when assembling submissions |

In PowerShell, use `$env:VARIABLE = "C:\path\to\input"`.
Place additional inputs under the repository root:

```text
data/
  test_answer_key.json                  # optional post-hoc scoring
  fewshot_config.json                  # required by the LLM judge
realtime/data/
  FinArg3_Social_Media_Pairwise_<date>.json
  FinArg3_Social_Media_Post_Ranking_2100_Collection.json
```

The judge's five few-shot pairs come from the restricted training release.
Supply `fewshot_config.json` with `posts` (five objects containing `a` and `b`
post texts) and `exclude_snippets` (distinctive substrings identifying exemplar
posts). Labels, rationales, and confidences are already in the code; registered
participants can request the exact config or post IDs from the authors.

## Workflows

Modules are grouped by purpose:

```text
finarg3_sm/
  preprocessing/    # normalization, grouped folds, lexicon features
  training/         # LR, encoders, auxiliary tasks, attribute model
  judging/          # LLM judge and attribute/ranking extraction
  inference/        # submissions, real-time scoring, top-210 ranking
  analysis/         # price data, stance audits, statistical analyses
  paths.py          # project-root paths and external input configuration
tests/
```

Generated artifacts stay in root-level `data/`, `models/`, `results/`,
`realtime/`, and `submissions/`. Run modules with `python -m`.

### Train and assemble submissions

```bash
python -m finarg3_sm.preprocessing.data_prep
python -m finarg3_sm.training.baseline_lr
python -m finarg3_sm.training.prefinetune

# Grouped CV; repeat with --seed 42 through 46.
python -m finarg3_sm.training.train_encoder --model models/macbert_dur --tag durpft --seed 42

# Train final checkpoints for each seed (42–46).
python -m finarg3_sm.training.train_encoder --model models/macbert_dur --tag durpft --seed 42 --final --save_model

python -m finarg3_sm.judging.llm_judge --mode cv --tag v3
python -m finarg3_sm.judging.llm_judge --mode test --tag v3
python -m finarg3_sm.judging.extract_attributes
python -m finarg3_sm.training.fit_attributes --attrs data/attrs_v2_gpt-5-mini.json
python -m finarg3_sm.inference.make_submission
```

Submission assembly uses `encoder_final_durpft_*.json`, judge outputs tagged
`v3`, and extracted attributes. It writes the three runs to `submissions/`.
Real-time inference also requires the saved encoder checkpoints.

### Real-time evaluation

```bash
python -m finarg3_sm.inference.predict_realtime --input "realtime/data/FinArg3_Social_Media_Pairwise_<date>.json"
python -m finarg3_sm.judging.extract_ranking
python -m finarg3_sm.inference.score_collection
python -m finarg3_sm.inference.build_top210
```

Replace `<date>` with the batch date. The last three commands build the
ranking submission from the 2,100-post collection.

### Post-hoc analysis

| Analysis | Modules (prefix each with `finarg3_sm.`) |
|---|---|
| Stance-corrected judge | `judging.make_v3sa`, then `judging.llm_judge_v3sa --mode test --tag v3sa` (`--mode cv` for CV) |
| July price bars | `analysis.fetch_ohlc` |
| Volatility–horizon scaling and gap statistics | `analysis.analyze_scaling` |
| Ex-ante capacity | `analysis.exante_capacity` |
| Text vs. capacity | `analysis.analyze_crossperiod`, `analysis.text_capacity_decomposition` |
| Stance-definition audit | `analysis.verify_stance_anchors` |
| Reliability figure | `analysis.plot_gap_reliability` (create `paper/` first) |

`analysis.collection_sample` supplies the shared July sample. Generate the
v3sa judge before running it; it remains a separate post-hoc variant.
Ablations use `judging.llm_judge --knn`, `training.train_encoder --soft_tau`
or `--weight_by_dmpp`, `training.prefinetune_rel`, and `training.embed_ranker`.

## Validation and reproducibility

Run the data-independent regression checks:

```bash
python -m unittest discover -s tests -p 'test_structure.py' -v
```

Optional: after generating ex-ante outputs, install `pytest` and run
`python -m pytest tests/test_exante_capacity.py -q`. These checks require the
analysis CSV, cached price history, and shared collection inputs.

Encoder seeds are 42–46; ex-ante analysis uses seed 42 and 20-/60-day lookbacks.
Complete LLM cache hits avoid new API calls; cache misses may incur charges and
produce different responses. Fixed seeds do not guarantee identical results
across environments. Yahoo Finance history may be restated; July analyses
truncate prices to 2026-07-24.

## Citation

The manuscript is unpublished:

```bibtex
@unpublished{baibaichuchu2026finarg3,
  author = {Zong-Han Bai and Po-Yen Chu},
  title  = {{BAIBAICHUCHU} at the {NTCIR}-19 {FinArg}-3 Task: When Is
            Maximum Possible Profit Predictable from Investor Text?},
  note   = {Unpublished manuscript},
  year   = {2026}
}
```
