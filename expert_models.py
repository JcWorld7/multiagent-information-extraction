"""Fine-tuned expert model loading and token-to-span inference."""
from __future__ import annotations

import json
from json import JSONDecodeError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from Labelcentered.settings import ID2TYPE, LABELS, TYPE2ID, ExpertSpec, PipelineSettings


@dataclass
class RawCandidate:
    label: str
    text: str
    start: int
    end: int
    expert_confidence: float
    expert_source: str
    threshold_used: float
    token_probabilities: List[float]
    source_section: str
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_thresholds(
    expert: ExpertSpec,
    allow_default_thresholds: bool = False,
) -> Dict[str, float]:
    path = expert.resolved_thresholds_path()
    if not path.exists():
        if not allow_default_thresholds:
            raise FileNotFoundError(
                f"Missing threshold file for {expert.name}: {path}. "
                "Pass --allow_default_thresholds to use 0.5 explicitly."
            )
        return {label: 0.5 for label in LABELS}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid threshold JSON for {expert.name} at {path}: {exc}") from exc
    missing = [label for label in LABELS if label not in payload]
    if missing:
        raise ValueError(f"{path} is missing thresholds for labels: {missing}")
    return {label: float(payload[label]) for label in LABELS}


def source_section_for_offset(source_text: str, start: int) -> str:
    methods = source_text.find("=== METHODS ===")
    results = source_text.find("=== RESULTS ===")
    if results != -1 and start >= results:
        return "results"
    if methods != -1 and start >= methods:
        return "methods"
    return "unknown"


class BertForMultiLabelTokenClassification:
    """Training-time model shape, with heavyweight imports delayed to IDAS runtime."""

    def __init__(self, model_name: str, num_types: int):
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = AutoConfig.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name, config=self.config)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.config.hidden_size, num_types)
                self.register_buffer("pos_weight", torch.ones(num_types))

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                x = self.dropout(out.last_hidden_state)
                return {"logits": self.classifier(x)}

        self.module = _Model()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> "BertForMultiLabelTokenClassification":
        self.module.to(*args, **kwargs)
        return self

    def eval(self) -> "BertForMultiLabelTokenClassification":
        self.module.eval()
        return self

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.load_state_dict(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.state_dict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


class ExpertExtractor:
    def __init__(
        self,
        spec: ExpertSpec,
        settings: PipelineSettings,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoTokenizer

        self.spec = spec
        self.settings = settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.thresholds = load_thresholds(spec, settings.allow_default_thresholds)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.base_encoder)
        self.model = BertForMultiLabelTokenClassification(spec.base_encoder, len(LABELS))
        self._load_weights(spec.checkpoint_dir)
        self.model.to(self.device).eval()

    def _load_weights(self, checkpoint_dir: Path) -> None:
        import torch

        safetensors_path = checkpoint_dir / "model.safetensors"
        bin_path = checkpoint_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path))
        elif bin_path.exists():
            state = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}"
            )
        self.model.load_state_dict(state, strict=False)

    def token_probs(self, source_text: str):
        import numpy as np
        import torch

        enc = self.tokenizer(
            source_text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.settings.max_len,
            stride=self.settings.stride,
            padding="max_length",
        )
        input_ids = torch.tensor(enc["input_ids"])
        attention = torch.tensor(enc["attention_mask"])
        all_probs = []
        with torch.no_grad():
            for i in range(0, input_ids.size(0), self.settings.infer_batch):
                ids = input_ids[i : i + self.settings.infer_batch].to(self.device)
                am = attention[i : i + self.settings.infer_batch].to(self.device)
                logits = self.model(input_ids=ids, attention_mask=am)["logits"]
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        return enc["offset_mapping"], enc["special_tokens_mask"], np.concatenate(all_probs, axis=0)

    def extract(self, source_text: str) -> List[RawCandidate]:
        offsets, special, probs = self.token_probs(source_text)
        intervals: Dict[int, List[Tuple[int, int, float]]] = {i: [] for i in range(len(LABELS))}
        token_probs_by_interval: Dict[Tuple[int, int, int], List[float]] = {}

        for chunk_index in range(probs.shape[0]):
            for token_index, ((start, end), special_mask) in enumerate(
                zip(offsets[chunk_index], special[chunk_index])
            ):
                if special_mask == 1 or end <= start:
                    continue
                token_stream = probs[chunk_index, token_index]
                for label_index, prob in enumerate(token_stream):
                    label = ID2TYPE[label_index]
                    if float(prob) >= self.thresholds[label]:
                        intervals[label_index].append((start, end, float(prob)))
                        token_probs_by_interval.setdefault((label_index, start, end), []).append(float(prob))

        candidates: List[RawCandidate] = []
        for label_index, label_intervals in intervals.items():
            label = ID2TYPE[label_index]
            for start, end, confidence, token_values in merge_intervals(label_intervals):
                while start < end and source_text[start].isspace():
                    start += 1
                while end > start and source_text[end - 1].isspace():
                    end -= 1
                text = source_text[start:end]
                if not text:
                    continue
                if source_text[start:end] != text:
                    raise AssertionError("Internal offset validation failed")
                candidates.append(
                    RawCandidate(
                        label=label,
                        text=text,
                        start=start,
                        end=end,
                        expert_confidence=round(confidence, 4),
                        expert_source=self.spec.name,
                        threshold_used=self.thresholds[label],
                        token_probabilities=[round(x, 6) for x in token_values],
                        source_section=source_section_for_offset(source_text, start),
                        provenance={
                            "expert": self.spec.name,
                            "base_encoder": self.spec.base_encoder,
                            "checkpoint_dir": str(self.spec.checkpoint_dir),
                        },
                    )
                )
        candidates.sort(key=lambda item: (item.label, item.start, item.end, item.text))
        return candidates


def merge_intervals(intervals: Sequence[Tuple[int, int, float]]) -> List[Tuple[int, int, float, List[float]]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged = []
    cur_start, cur_end = ordered[0][0], ordered[0][1]
    probs = [ordered[0][2]]
    for start, end, prob in ordered[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
            probs.append(prob)
        else:
            merged.append((cur_start, cur_end, sum(probs) / len(probs), list(probs)))
            cur_start, cur_end, probs = start, end, [prob]
    merged.append((cur_start, cur_end, sum(probs) / len(probs), list(probs)))
    return merged


class ExpertEnsemble:
    def __init__(self, settings: PipelineSettings, device: str | None = None):
        self.settings = settings
        self.extractors = [ExpertExtractor(spec, settings, device=device) for spec in settings.experts]

    def extract_by_expert_and_label(self, source_text: str) -> Dict[str, Dict[str, List[RawCandidate]]]:
        output: Dict[str, Dict[str, List[RawCandidate]]] = {}
        for extractor in self.extractors:
            by_label = {label: [] for label in LABELS}
            for candidate in extractor.extract(source_text):
                by_label[candidate.label].append(candidate)
            output[extractor.spec.name] = by_label
        return output


def thresholds_by_expert_and_label(settings: PipelineSettings) -> Dict[str, Dict[str, float]]:
    return {
        spec.name: load_thresholds(spec, settings.allow_default_thresholds)
        for spec in settings.experts
    }


def raw_candidates_to_json(raw: Mapping[str, Mapping[str, Sequence[RawCandidate]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    return {
        expert: {
            label: [candidate.to_dict() for candidate in candidates]
            for label, candidates in by_label.items()
        }
        for expert, by_label in raw.items()
    }
