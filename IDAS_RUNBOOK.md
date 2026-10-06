# Labelcentered IDAS Runbook

This document explains the label-centered PICO extraction workflow and how to
run it on UIowa IDAS after copying the `Labelcentered/` directory to the Agent
project.

The intended IDAS location is:

```bash
/home/zhf/hpchome/ModelForPICO/ExtractiveModel/Agent/Labelcentered
```

Run commands from the parent Agent directory:

```bash
cd /home/zhf/hpchome/ModelForPICO/ExtractiveModel/Agent
```

## What This Pipeline Does

The system performs strict extractive PICO extraction. It never allows an LLM to
directly write final span text or offsets.

For each source article:

1. The three fine-tuned expert models run token-level extraction:
   - SciBERT
   - PubMedBERT
   - ClinicalBERT

2. Each expert produces seven label streams:
   - Participant
   - Intervention
   - Outcome
   - SampleSize
   - ComparisonGroup
   - DesignDescription
   - StatisticalAnalysis

3. The 21 expert-label streams are converted into exact candidate spans.

4. Candidates are validated against the original source text:

   ```python
   source_text[start:end] == candidate_text
   ```

5. The registry builds seven shared label-specific candidate banks.

6. For each label, three DoctorAgents review the same candidate bank:
   - `SciBERT_<Label>_Doctor`
   - `PubMedBERT_<Label>_Doctor`
   - `ClinicalBERT_<Label>_Doctor`

7. Each doctor must return a candidate-level decision for every visible
   candidate:
   - `keep`
   - `reject`

8. The LabelMetaAgent compares the three doctor reviews for that label.

9. If the doctors disagree, or if the current candidates have bad boundaries or
   miss important text, the LabelMetaAgent may request an exact quote expansion.

10. The Registry Manager verifies every proposed quote against the original
    source text before adding it as a new immutable candidate ID.

11. All three doctors then review the expanded candidate bank again.

12. The process repeats until strict consensus is reached or the maximum number
    of rounds is reached.

13. The GlobalMetaAgent assembles the seven frozen label outputs and checks for
    cross-label conflicts.

14. Optional summaries may be generated only after exact extraction is frozen.
    Summaries are stored separately under `generated_summaries`.

## Strict Consensus Rule

Consensus is candidate-level and unanimous by default.

A selected candidate is accepted by consensus only when all three doctors vote:

```text
keep, keep, keep
```

A 2-to-1 vote is recorded as majority agreement, not consensus.

The system does not use `agree=true` booleans as the consensus criterion.

## Append-Only Candidate Registry

Candidate IDs are immutable.

Once a candidate is created, these fields must not change:

- `candidate_id`
- `label`
- `text`
- `start`
- `end`

If a candidate boundary is too long, too short, fragmented, or incomplete, the
pipeline creates a new candidate ID. It does not edit the old candidate.

Examples:

```text
OUT-C0001  original expert candidate
OUT-C0002  exact quote proposed during consultation and verified by registry
```

Every final span must come from the final candidate registry.

## Default IDAS Model Paths

The default checkpoint paths are configured in `settings.py`.

SciBERT:

```bash
/home/zhf/hpchome/ModelForPICO/ExtractiveModel/10times250filesSciBert/eight/checkpoint-1833
```

PubMedBERT:

```bash
/home/zhf/hpchome/ModelForPICO/ExtractiveModel/10times250filesPubMed/eight/checkpoint-2145
```

ClinicalBERT:

```bash
/home/zhf/hpchome/ModelForPICO/ExtractiveModel/10times250filesClinical/eight/checkpoint-1043
```

Each checkpoint directory must contain:

```bash
best_thresholds.json
```

The pipeline fails clearly when thresholds are missing. It does not silently use
0.5. For debugging only, you may pass:

```bash
--allow_default_thresholds
```

## Before Running On IDAS

After copying `Labelcentered/` to IDAS, confirm the package is visible:

```bash
cd /home/zhf/hpchome/ModelForPICO/ExtractiveModel/Agent
ls Labelcentered
```

Expected important files:

```text
run_pdf_labelcentered.py
run_json_labelcentered.py
settings.py
expert_models.py
consultation.py
output_validation.py
inspect_output.py
```

Check the threshold files:

```bash
ls /home/zhf/hpchome/ModelForPICO/ExtractiveModel/10times250filesSciBert/eight/checkpoint-1833/best_thresholds.json
ls /home/zhf/hpchome/ModelForPICO/ExtractiveModel/10times250filesPubMed/eight/checkpoint-2145/best_thresholds.json
ls /home/zhf/hpchome/ModelForPICO/ExtractiveModel/10times250filesClinical/eight/checkpoint-1043/best_thresholds.json
```

## Recommended IDAS Session Layout

Use two terminals or two jobs:

1. A vLLM server process using the L40S GPU.
2. A pipeline process that connects to the local vLLM endpoint.

The default endpoint is:

```text
http://127.0.0.1:8000/v1
```

The default served model is:

```text
Qwen/Qwen2.5-7B-Instruct
```

## Start vLLM On The L40S GPU

Activate the vLLM environment:

```bash
source /home/zhf/vllm-cu128/bin/activate
```

Start the OpenAI-compatible local server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.70 \
  --dtype auto
```

Keep this process running while the pipeline runs.

If IDAS requires an interactive GPU allocation before starting vLLM, request an
L40S GPU using the local IDAS scheduler command recommended by your cluster
documentation, then run the command above inside that allocation.

## Run The PDF Pipeline

In a second terminal or job, activate the pipeline environment:

```bash
source /home/zhf/myenv/bin/activate
```

Go to the Agent directory:

```bash
cd /home/zhf/hpchome/ModelForPICO/ExtractiveModel/Agent
```

Run one PDF:

```bash
python Labelcentered/run_pdf_labelcentered.py \
  --pdf ./Wong_635.pdf \
  --out_dir ./labelcentered_pdf_out \
  --save_text
```

Run all PDFs in a directory:

```bash
python Labelcentered/run_pdf_labelcentered.py \
  --pdf_dir ./pdfs \
  --out_dir ./labelcentered_pdf_out \
  --save_text \
  --skip_existing
```

Useful options:

```text
--classifier_only            Run expert models and registry construction only.
--save_text                  Retained for compatibility; PDF outputs now always include model_input_text.
--skip_existing              Do not rerun files that already have output.
--generate_summaries         Generate optional summaries after exact extraction.
--allow_default_thresholds   Explicitly use 0.5 if threshold files are missing.
--max_rounds 3               Maximum LabelMetaAgent consultation rounds.
--consensus_rule unanimous   Current strict consensus rule.
--final_selection_policy expert_consensus_verified
                             Default equal-expert final selection policy.
--disable_deterministic_pre_expansion
                             Ablation only. Disables source-grounded phrase and
                             boundary candidate generation before doctor review.
--doctor_batch_size 25       Number of candidates per doctor LLM call.
--llm_timeout_seconds 900    OpenAI-compatible request timeout; raise for slow vLLM generations.
```

## Run The JSON Pipeline

Use this for Label Studio-style JSON or configured dataset files.

Run one explicit JSON file:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json /path/to/file.json \
  --out_dir ./labelcentered_json_out \
  --limit_records 1
```

Run with classifier-only mode:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json /path/to/file.json \
  --out_dir ./labelcentered_json_out \
  --limit_records 1 \
  --classifier_only
```

## Output Files

Each processed input writes a JSON output in the selected `--out_dir`.

The output includes:

- `source_file`
- `source_path`
- `model_input_text`
- `expert_models`
- `thresholds_by_expert_and_label`
- `raw_candidates_by_expert_and_label`
- `cleaning_log`
- `label_candidate_registries`
- `registry_expansion_history`
- `doctor_reviews_by_label`
- `review_completeness_by_label`
- `consistency_metrics_by_label`
- `candidate_vote_matrices_by_label`
- `rounds_by_label`
- `label_meta_decisions`
- `label_consensus_status`
- `label_termination_reasons`
- `global_meta_decision`
- `final_candidate_ids_by_label`
- `final_spans_by_label`
- `final_spans`
- `missing_labels`
- `unresolved_candidates`
- `cross_label_conflicts`
- `overall_consensus_reached`
- `generated_summaries` when requested

## Inspect Output

Print readable spans, votes, consensus status, and conflicts:

```bash
python Labelcentered/inspect_output.py \
  ./labelcentered_pdf_out/Wong_635.labelcentered.json
```

If the output filename differs, list the directory:

```bash
ls ./labelcentered_pdf_out
```

## Validate Output

Run validation after extraction:

```bash
python Labelcentered/output_validation.py \
  ./labelcentered_pdf_out/Wong_635.labelcentered.json
```

The validator checks:

- all labels are valid;
- all seven labels exist;
- all final candidate IDs exist in the registry;
- final offsets are valid;
- final span text exactly matches `model_input_text`;
- `final_spans_by_label` agrees with `final_spans`;
- complete reviews are actually complete;
- consensus is not true when unresolved candidate decisions remain;
- generated summaries do not replace exact spans.

The command exits nonzero if validation fails.

## Calibrate Final Selection From Annotated Data

Calibration is offline. Do not rerun BERT and LLM agents for every calibration
setting. Run the pipeline once, cache candidates/registries/doctor votes, then
tune final selection rules from the compact cache.

First create record-level splits from `settings.DEFAULT_DATASET_PATHS`:

```bash
python Labelcentered/make_labelcentered_calibration_splits.py \
  --out_dir ./labelcentered_calibration_splits \
  --dev_small 50 \
  --dev_check 50
```

Recommended use:

```text
dev_small   30-50 records    fast calibration
dev_check   50 records       stability confirmation
test        remaining        final evaluation only
```

Run the expensive pipeline once for the small dev split:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json ./labelcentered_calibration_splits/dev_small.json \
  --out_dir ./labelcentered_dev_small_predictions \
  --doctor_batch_size 10 \
  --max_rounds 3 \
  --llm_temperature 0
```

Build the compact cache:

```bash
python Labelcentered/build_labelcentered_calibration_cache.py \
  --gold ./labelcentered_calibration_splits/dev_small.json \
  --pred_dir ./labelcentered_dev_small_predictions \
  --out_json ./labelcentered_dev_small.cache.json \
  --variant dev_small
```

If you already built a cache before `recommended_candidate_ids_by_label` was
stored, rebuild it once. The cache builder is fast and does not rerun BERT or
LLM agents.

Tune from the cache. This step is fast and can be repeated without rerunning the
pipeline:

```bash
python Labelcentered/tune_labelcentered_calibration.py \
  --cache_json ./labelcentered_dev_small.cache.json \
  --out_json ./labelcentered_calibration.json \
  --prefer f1
```

The tuner can now choose fixed per-label source modes such as `recommended`,
`expert_consensus_verified`, `pubmedbert_plus_verified`, and `registry_filtered`,
or the lightweight calibrated rule. Use `--allow_registry_all_mode` only for an
experiment; it can overfit small dev sets because it selects every registry
candidate.

To check stability, run/build a second cache for `dev_check.json` and tune or
evaluate against it. To test boundary normalization, run the same split twice,
build two caches, and pass both cache files:

```bash
python Labelcentered/tune_labelcentered_calibration.py \
  --cache_json ./dev_small_base.cache.json ./dev_small_normalized.cache.json \
  --out_json ./labelcentered_calibration.json \
  --prefer f1
```

Apply a learned config to existing outputs without rerunning extraction:

```bash
python Labelcentered/apply_labelcentered_calibration.py \
  --config ./labelcentered_calibration.json \
  --pred_dir ./labelcentered_test_predictions \
  --out_dir ./labelcentered_test_calibrated
```

Use a learned config directly during new extraction runs:

```bash
python Labelcentered/run_json_labelcentered.py \
  --json ./Labelcentered/new_data.json \
  --out_dir ./labelcentered_new_calibrated \
  --final_selection_policy calibrated \
  --calibration_config ./labelcentered_calibration.json \
  --doctor_batch_size 10 \
  --max_rounds 3 \
  --llm_temperature 0
```

## Local CPU Checks Before Copying

On the laptop, do not load CUDA, checkpoints, vLLM, or the real PDF. Only run
mocked checks:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/labelcentered_pycache \
python3 -m py_compile Labelcentered/*.py
```

```bash
PYTHONPYCACHEPREFIX=/private/tmp/labelcentered_pycache \
python3 -m unittest discover \
  -s Labelcentered/tests \
  -p "test_*.py"
```

## Expected LLM Calls

The number of calls depends on candidate counts and disagreement.

For each label and each round:

```text
3 doctor roles x ceil(number_of_label_candidates / doctor_batch_size)
+ 1 LabelMetaAgent call
```

Across seven labels and three rounds, the worst-case approximate count is:

```text
7 x 3 x ceil(candidates_per_label / doctor_batch_size) doctor calls
+ 7 x 3 LabelMetaAgent calls
```

If summaries are enabled, add one SummaryAgent call.

Registry expansion may increase candidates in later rounds, which can increase
doctor calls.

## Context Management

The pipeline avoids sending the full document in every LLM prompt.

It uses:

- label-specific candidate banks;
- doctor review batches;
- candidate-local context windows;
- label-specific RAG guidelines;
- sentence/source-unit IDs for exact quote proposals;
- structured JSON output from vLLM.

The vLLM request uses:

```python
extra_body={
    "structured_outputs": {
        "json": schema
    }
}
```

It does not use deprecated `guided_json`.

## How Disagreement Repair Works

For a given label:

1. The registry provides the current label candidate bank.
2. All three doctors review the same candidate IDs.
3. The LabelMetaAgent examines:
   - doctor decisions;
   - vote matrix;
   - label-specific consistency metrics;
   - candidate-local source context;
   - relevant sentence/source-unit chunks.
4. If current candidates are inadequate, the LabelMetaAgent requests expansion,
   usually with an exact quote and `sentence_id`.
5. The Registry Manager searches the original source text.
6. If the quote is exact and unambiguous, the registry appends a new candidate.
7. The new candidate is not final yet.
8. In the next round, all three doctors review the expanded candidate bank.
9. The LabelMetaAgent can select the new candidate only if doctor votes support
   it.
10. Consensus requires unanimous keep votes for every selected candidate.

This protects the final output from LLM paraphrases and generated offsets.

## Common Failure Modes

### vLLM connection refused

Check that vLLM is still running:

```bash
curl http://127.0.0.1:8000/v1/models
```

If this fails, restart vLLM.

### Missing threshold file

The pipeline will fail with a message naming the missing `best_thresholds.json`.
Either place the threshold file in the checkpoint directory or rerun explicitly
with:

```bash
--allow_default_thresholds
```

Use that fallback only for debugging.

### CUDA out of memory

Try one or more of:

```text
lower --infer_batch
lower vLLM --gpu-memory-utilization
lower vLLM --max-model-len
reduce --doctor_batch_size
```

### Too many candidates in one prompt

Lower:

```bash
--doctor_batch_size 10
```

### Consensus is false

This is allowed. The system should not force consensus.

Inspect:

```text
label_consensus_status
label_termination_reasons
unresolved_candidates
rounds_by_label
candidate_vote_matrices_by_label
registry_expansion_history
```

### Final validation fails on offsets

Do not manually fix offsets in output JSON. Inspect the source text and registry
candidate that failed. The correct fix is to adjust candidate generation or
expansion logic and rerun.

## Recommended First IDAS Test

First run classifier-only to confirm checkpoint loading and threshold files:

```bash
python Labelcentered/run_pdf_labelcentered.py \
  --pdf ./Wong_635.pdf \
  --out_dir ./labelcentered_classifier_only_test \
  --save_text \
  --classifier_only
```

Then run the full agent workflow with vLLM running:

```bash
python Labelcentered/run_pdf_labelcentered.py \
  --pdf ./Wong_635.pdf \
  --out_dir ./labelcentered_pdf_out \
  --save_text \
  --doctor_batch_size 25 \
  --max_rounds 3
```

Validate:

```bash
python Labelcentered/output_validation.py \
  ./labelcentered_pdf_out/Wong_635.labelcentered.json
```

Inspect:

```bash
python Labelcentered/inspect_output.py \
  ./labelcentered_pdf_out/Wong_635.labelcentered.json
```

## What Still Requires IDAS Testing

Laptop tests are mocked and CPU-only. IDAS testing is required for:

- loading all three Hugging Face base encoders;
- loading fine-tuned checkpoints;
- loading per-model `best_thresholds.json`;
- CUDA inference on L40S;
- PDF extraction from the real article;
- vLLM structured JSON compatibility;
- prompt size behavior on long PDFs;
- full consultation runtime and LLM output quality;
- output validation on real extracted documents.


Before DoctorAgent review, the registry now performs deterministic
source-grounded pre-expansion for Participant, Intervention, Outcome, and
SampleSize. It adds exact cue-centered phrases, number-centered sample spans,
and boundary alternatives from broad candidates. Disable it only for ablation
with `--disable_deterministic_pre_expansion`.

DoctorAgents also receive compact source sentence IDs and their exact-quote
repair proposals are now sent to the expansion manager. Inspect
`doctor_expansion_requests`, `label_meta_expansion_requests`, and
`proposal_outcomes` in output JSON when checking whether agents are proposing
valid new spans.
