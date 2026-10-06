import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.evaluate_labelcentered import (
    EvalSpan,
    evaluate,
    readable_report,
    token_assignment_accuracy,
    token_level_metrics,
    tokenize_with_offsets,
)
from Labelcentered.settings import LABELS


class EvaluatorTests(unittest.TestCase):
    def make_gold(self):
        method = "Participants included 63 students.\nOutcome score improved."
        source = method + "\n\n"
        participant = "63 students"
        outcome = "Outcome score"
        participant_start = source.index(participant)
        outcome_start = source.index(outcome)
        results = []
        for label, text, start in [
            ("Participant", participant, participant_start),
            ("Outcome", outcome, outcome_start),
        ]:
            results.append({
                "type": "labels",
                "value": {
                    "start": start,
                    "end": start + len(text),
                    "text": text,
                    "labels": [label],
                },
            })
        return [{
            "id": 1,
            "data": {
                "file": "paper.pdf",
                "method_section": method,
                "results_section": "",
            },
            "annotations": [{"result": results}],
        }]

    def make_prediction(self):
        labels = {label: [] for label in LABELS}
        ids = {label: [] for label in LABELS}
        registries = {label: [] for label in LABELS}
        source = "Participants included 63 students.\nOutcome score improved.\n\n"
        participant_start = source.index("63 students")
        outcome_start = source.index("Outcome score")
        participant = {
            "candidate_id": "PAR-C0001",
            "label": "Participant",
            "text": "63 students",
            "start": participant_start,
            "end": participant_start + len("63 students"),
            "proposed_by_experts": ["PubMedBERT"],
        }
        relaxed_outcome = {
            "candidate_id": "OUT-C0001",
            "label": "Outcome",
            "text": "Outcome score improved",
            "start": outcome_start,
            "end": outcome_start + len("Outcome score improved"),
            "proposed_by_experts": ["ClinicalBERT"],
        }
        labels["Participant"] = [participant]
        labels["Outcome"] = [relaxed_outcome]
        ids["Participant"] = ["PAR-C0001"]
        ids["Outcome"] = ["OUT-C0001"]
        registries["Participant"] = [participant]
        registries["Outcome"] = [
            relaxed_outcome,
            {
                "candidate_id": "OUT-C0002",
                "label": "Outcome",
                "text": "Outcome score",
                "start": outcome_start,
                "end": outcome_start + len("Outcome score"),
                "proposed_by_experts": ["PubMedBERT"],
            },
        ]
        return {
            "record_index": 0,
            "record_name": "paper.pdf",
            "input_type": "json",
            "model_input_text": source,
            "final_spans_by_label": labels,
            "final_candidate_ids_by_label": ids,
            "recommended_candidate_ids_by_label": ids,
            "label_candidate_registries": registries,
            "label_consensus_status": {label: label == "Participant" for label in LABELS},
            "candidate_vote_matrices_by_label": {
                "Participant": {
                    "PAR-C0001": {
                        "SciBERT_Participant_Doctor": "reject",
                        "PubMedBERT_Participant_Doctor": "keep",
                        "ClinicalBERT_Participant_Doctor": "reject",
                    }
                },
                "Outcome": {
                    "OUT-C0001": {
                        "SciBERT_Outcome_Doctor": "keep",
                        "PubMedBERT_Outcome_Doctor": "keep",
                        "ClinicalBERT_Outcome_Doctor": "reject",
                    },
                    "OUT-C0002": {
                        "SciBERT_Outcome_Doctor": "reject",
                        "PubMedBERT_Outcome_Doctor": "keep",
                        "ClinicalBERT_Outcome_Doctor": "reject",
                    },
                },
            },
            "overall_consensus_reached": False,
        }

    def test_exact_relaxed_and_registry_oracle_metrics(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            gold_path = directory / "gold.json"
            prediction_path = directory / "prediction.labelcentered.json"
            gold_path.write_text(json.dumps(self.make_gold()), encoding="utf-8")
            prediction_path.write_text(json.dumps(self.make_prediction()), encoding="utf-8")
            report = evaluate(gold_path, [prediction_path], [0.5, 0.75])

        self.assertEqual(report["aligned_record_count"], 1)
        self.assertEqual(report["final_exact_span_metrics"]["Participant"]["tp"], 1)
        self.assertEqual(report["final_exact_span_metrics"]["Outcome"]["tp"], 0)
        self.assertEqual(report["final_relaxed_span_metrics"]["0.5"]["Outcome"]["tp"], 1)
        self.assertEqual(report["registry_oracle_exact_recall"]["Outcome"]["recall"], 1.0)
        self.assertGreater(report["character_level_metrics"]["Outcome"]["f1"], 0)
        self.assertGreater(
            report["character_label_confusion"]["Participant"]["Participant"],
            0,
        )
        token = report["final_token_level_metrics"]
        self.assertEqual(token["Participant"]["tp"], 2)
        self.assertEqual(token["Outcome"]["tp"], 2)
        self.assertEqual(token["Outcome"]["fp"], 1)
        self.assertGreater(token["micro"]["f1"], 0)
        self.assertIn("accuracy", token["micro"])
        self.assertIn("accuracy", token["Participant"])
        self.assertIn("final_token_assignment_accuracy", report)
        self.assertGreater(report["final_token_assignment_accuracy"]["accuracy"], 0)
        self.assertIn("labeled_accuracy", report["final_token_assignment_accuracy"])
        self.assertIn("Labeled-token assignment accuracy", readable_report(report))
        self.assertIn("token_label_confusion", report)
        self.assertEqual(
            report["registry_oracle_token_recall"]["Outcome"]["recall"],
            1.0,
        )
        ablations = report["selection_ablation_metrics"]
        self.assertIn("pubmedbert_only", ablations)
        self.assertEqual(ablations["pubmedbert_only"]["token_level_metrics"]["Participant"]["tp"], 2)
        self.assertEqual(ablations["majority_vote"]["token_level_metrics"]["Outcome"]["tp"], 2)
        self.assertIn("SELECTION ABLATION", readable_report(report))
        self.assertIn("TOKEN-LEVEL METRICS", readable_report(report))
        self.assertIn("Micro F1", readable_report(report))

    def test_annotation_text_mismatch_repairs_flexible_whitespace(self):
        gold = self.make_gold()
        gold[0]["annotations"][0]["result"][0]["value"]["text"] = "63\\nstudents"
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            gold_path = directory / "gold.json"
            prediction_path = directory / "prediction.labelcentered.json"
            gold_path.write_text(json.dumps(gold), encoding="utf-8")
            prediction_path.write_text(json.dumps(self.make_prediction()), encoding="utf-8")
            report = evaluate(gold_path, [prediction_path])
        self.assertEqual(report["annotation_text_mismatch_count"], 0)
        self.assertEqual(report["gold_annotation_repair_count"], 1)
        self.assertEqual(report["final_exact_span_metrics"]["Participant"]["tp"], 1)

    def test_tokenizer_preserves_offsets_and_macro_metrics(self):
        text = "self-esteem improved; n=63."
        tokens = tokenize_with_offsets(text)
        self.assertEqual(
            [text[token.start:token.end] for token in tokens],
            ["self-esteem", "improved", ";", "n", "=", "63", "."],
        )
        gold = [
            EvalSpan("Outcome", 0, len("self-esteem"), "self-esteem"),
            EvalSpan("SampleSize", text.index("63"), text.index("63") + 2, "63"),
        ]
        predicted = [
            EvalSpan("Outcome", 0, len("self-esteem improved"), "self-esteem improved"),
            EvalSpan("SampleSize", text.index("63"), text.index("63") + 2, "63"),
        ]
        metrics = token_level_metrics([(tokens, gold, predicted)])
        self.assertEqual(metrics["SampleSize"]["f1"], 1.0)
        self.assertLess(metrics["Outcome"]["precision"], 1.0)
        self.assertIn("micro", metrics)
        self.assertIn("macro", metrics)
        self.assertIn("accuracy", metrics["Outcome"])
        assignment = token_assignment_accuracy([(tokens, gold, predicted)])
        self.assertLess(assignment["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
