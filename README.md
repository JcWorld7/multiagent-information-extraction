# Labelcentered PICO Extraction

This package is a self-contained label-centered rewrite. It does not import the
old `agents.py`, `strict_extractive`, or previous generated agent packages.

## Architecture

- Three fine-tuned expert models: SciBERT, PubMedBERT, ClinicalBERT.
- Seven labels in fixed order: Participant, Intervention, Outcome, SampleSize,
  ComparisonGroup, DesignDescription, StatisticalAnalysis.
- 3 x 7 = 21 expert-label probability streams.
- Seven shared label-specific candidate banks.
- 21 DoctorAgent roles, three per label.
- Seven LabelMetaAgents, one per label.
- One GlobalMetaAgent for frozen cross-label assembly.
- Optional SummaryAgent after exact extraction is frozen.

All final spans come from validated candidate IDs. DoctorAgents and MetaAgents
must not write final span text or offsets.

## Consensus

Consensus is candidate-level and unanimous by default. A candidate selected for a
label is accepted by consensus only when all three label doctors vote `keep`.
A 2-to-1 vote is recorded as majority agreement, not unanimous consensus.

Consistency metrics include pairwise raw agreement, Cohen's kappa, keep-set
Jaccard, three-way unanimous agreement rate, Fleiss' kappa, vote matrices, and
conflicted IDs.

## Registry Expansion

The registry is append-only. Exact duplicate expert candidates merge provenance;
new alternatives receive new label-aware IDs such as `OUT-C0001` or `STA-C0001`.
Candidate text, label, start, and end are immutable after creation.

Agents may request expansion with sentence IDs, exact quotes, or controlled
operations. The deterministic expansion manager validates every new candidate
against the source text before appending it. DoctorAgent exact-quote proposals
are now executed too, not only LabelMetaAgent proposals, so boundary repairs
found during review enter the registry for the next round.

## IDAS vLLM

For the full workflow, output interpretation, validation commands, and IDAS
troubleshooting, see `IDAS_RUNBOOK.md`.

Start one backend at a time. The selected `--llm_backend` must match the model
served by vLLM at `--llm_base_url`.

```bash
source /home/zhf/vllm-cu128/bin/activate

# Qwen
Labelcentered/scripts/start_vllm_backend.sh qwen

# Llama 3.1 8B. Use HF_TOKEN or run huggingface-cli login first.
Labelcentered/scripts/start_vllm_backend.sh llama31_8b

# Mistral 7B
Labelcentered/scripts/start_vllm_backend.sh mistral7b
```

The launcher uses `--served-model-name` with the exact Hugging Face model ID so
OpenAI-compatible requests do not fail with a model-name 404.

## IDAS Pipeline

```bash
source /home/zhf/myenv/bin/activate

cd /home/zhf/hpchome/ModelForPICO/ExtractiveModel/Agent

python Labelcentered/run_pdf_labelcentered.py \
  --pdf ./Wong_635.pdf \
  --out_dir ./labelcentered_pdf_out \
  --save_text
```

## PDF Run

```bash
python Labelcentered/run_pdf_labelcentered.py \
  --pdf ./Wong_635.pdf \
  --out_dir ./labelcentered_pdf_out \
  --save_text
```

## JSON Run

```bash
python Labelcentered/run_json_labelcentered.py \
  --json /path/to/file.json \
  --out_dir ./labelcentered_json_out \
  --limit_records 1
```

## Inspection And Validation

```bash
python Labelcentered/inspect_output.py ./labelcentered_pdf_out/article.labelcentered.json
python Labelcentered/output_validation.py ./labelcentered_pdf_out/article.labelcentered.json
```

## Human-Annotated Evaluation

Run predictions for the 10-record Label Studio gold file:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json ./Labelcentered/Ex_161_170_Feb.16_2026.json \
  --out_dir ./labelcentered_eval_predictions \
  --doctor_batch_size 10 \
  --max_rounds 3 \
  --llm_context_window 16384 \
  --llm_max_tokens 2048 \
  --llm_timeout_seconds 1800
```

Evaluate the prediction directory:

```bash
python Labelcentered/evaluate_labelcentered.py \
  --gold ./Labelcentered/Ex_161_170_Feb.16_2026.json \
  --pred_dir ./labelcentered_eval_predictions \
  --out_json ./labelcentered_evaluation.json \
  --out_txt ./labelcentered_evaluation.txt
```

The evaluator reports per-label, micro, and macro exact-span precision, recall,
and F1; relaxed span F1 at configurable IoU thresholds; deterministic
offset-preserving token-level precision, recall, F1, and label confusion;
character-level metrics; document-level label-presence confusion; boundary
errors; matched-span label confusion; consensus coverage; and candidate-registry
oracle span and token recall.

The default final selection policy is `expert_consensus_verified`, which treats
SciBERT, PubMedBERT, and ClinicalBERT equally. Keep `strict` and
`pubmedbert_plus_verified` for baseline/ablation runs.

Before DoctorAgent review, the registry now performs deterministic
source-grounded pre-expansion for Participant, Intervention, Outcome, and
SampleSize. It adds exact cue-centered phrases, number-centered sample spans,
and boundary alternatives from broad candidates. Disable it only for ablation
with `--disable_deterministic_pre_expansion`.

## Calibration From Annotated Data

Create fast record-level splits from the default IDAS annotated data paths:

```bash
python Labelcentered/make_labelcentered_calibration_splits.py \
  --out_dir ./labelcentered_calibration_splits \
  --dev_small 50 \
  --dev_check 50
```

Run the expensive BERT + LLM pipeline once on each split you want to evaluate:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json ./labelcentered_calibration_splits/dev_small.json \
  --out_dir ./labelcentered_dev_small_predictions \
  --doctor_batch_size 10 \
  --max_rounds 3 \
  --llm_temperature 0
```

Build a compact offline cache from the saved candidates, registries, and doctor
votes:

```bash
python Labelcentered/build_labelcentered_calibration_cache.py \
  --gold ./labelcentered_calibration_splits/dev_small.json \
  --pred_dir ./labelcentered_dev_small_predictions \
  --out_json ./labelcentered_dev_small.cache.json \
  --variant dev_small
```

Rebuild old caches once so they include `recommended_candidate_ids_by_label`;
this is quick and does not rerun BERT or LLM agents.

Tune final selection rules from the cache without rerunning BERT or LLM agents:

```bash
python Labelcentered/tune_labelcentered_calibration.py \
  --cache_json ./labelcentered_dev_small.cache.json \
  --out_json ./labelcentered_calibration.json \
  --prefer f1
```

The tuner can choose per-label fixed modes such as `recommended` and
`registry_filtered`, plus the lightweight calibrated rule. `registry_all` is
available only with `--allow_registry_all_mode` because it can overfit small
calibration sets.

Apply the learned selector to existing prediction payloads without rerunning
BERT or the LLM agents:

```bash
python Labelcentered/apply_labelcentered_calibration.py \
  --config ./labelcentered_calibration.json \
  --pred_dir ./labelcentered_test_predictions \
  --out_dir ./labelcentered_test_calibrated
```

Use it directly for new runs:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json ./Labelcentered/new_data.json \
  --out_dir ./labelcentered_new_calibrated \
  --final_selection_policy calibrated \
  --calibration_config ./labelcentered_calibration.json
```

To evaluate boundary normalization, run predictions twice, build two caches, and
pass both cache files to the tuner. The selected variant is written into the
calibration config.

Gold `start`, `end`, and label values are authoritative. Label Studio annotation
text differences caused by escaped newlines or whitespace are recorded as
diagnostics without changing the gold offsets.

## Thresholds

Each checkpoint must have `best_thresholds.json`. Missing thresholds fail by
default. Use `--allow_default_thresholds` only for explicit debugging fallback to
0.5.

## Known Limitations

- Local laptop tests use synthetic text and mocked LLM/model outputs.
- Actual checkpoint loading, CUDA behavior, vLLM prompts, and PDF extraction
  require IDAS integration testing.
- Label-specific isolation still needs GlobalMetaAgent checks for cross-label
  overlap and exact duplicates.
