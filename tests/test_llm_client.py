import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.llm_client import (
    LLMClient,
    completion_token_budget,
    context_limits_from_error,
    parse_json_object,
)
from Labelcentered.settings import PipelineSettings
from Labelcentered.schemas import (
    LABEL_META_SCHEMA,
    doctor_review_schema,
    label_meta_schema,
)


class LLMClientParsingTests(unittest.TestCase):

    def test_validate_model_available_accepts_served_model(self):
        settings = PipelineSettings(
            llm_backend="llama31_8b",
            llm_model="meta-llama/Llama-3.1-8B-Instruct",
        )
        client = LLMClient(settings)
        client.available_models = lambda: ["meta-llama/Llama-3.1-8B-Instruct"]
        client.validate_model_available()

    def test_validate_model_available_explains_mismatch(self):
        settings = PipelineSettings(
            llm_backend="llama31_8b",
            llm_model="meta-llama/Llama-3.1-8B-Instruct",
        )
        client = LLMClient(settings)
        client.available_models = lambda: ["Qwen/Qwen2.5-7B-Instruct"]
        with self.assertRaisesRegex(RuntimeError, "scripts/start_vllm_backend.sh llama31_8b"):
            client.validate_model_available()

    def test_openai_client_uses_configured_timeout(self):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        original = sys.modules.get("openai")
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            client = LLMClient(PipelineSettings(llm_timeout_seconds=1234.5))
            client._openai_client()
        finally:
            if original is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = original

        self.assertEqual(captured["timeout"], 1234.5)



    def test_structured_chat_falls_back_without_vllm_structured_outputs(self):
        client = LLMClient(PipelineSettings())
        structured_modes = []

        def fake_chat(model, system, user, schema, source_name, use_structured_outputs=True):
            structured_modes.append(use_structured_outputs)
            if use_structured_outputs:
                return '{"selected_candidate_ids": ['
            return '{"selected_candidate_ids": [], "rationale": "ok"}'

        client._chat_completion = fake_chat
        parsed = client.structured_chat(
            system="system",
            user="user",
            schema={"type": "object", "properties": {}},
            call_name="TestAgent",
        )
        self.assertEqual(parsed["selected_candidate_ids"], [])
        self.assertTrue(parsed["_used_unstructured_json_fallback"])
        self.assertEqual(structured_modes, [True, True, False])

    def test_parse_markdown_wrapped_json(self):
        parsed = parse_json_object('```json\n{"decisions": []}\n```')
        self.assertEqual(parsed, {"decisions": []})

    def test_parse_json_with_trailing_comma(self):
        parsed = parse_json_object('{"selected_candidate_ids": ["OUT-C0001"],}')
        self.assertEqual(parsed["selected_candidate_ids"], ["OUT-C0001"])

    def test_parse_json_with_surrounding_text(self):
        parsed = parse_json_object('Here is the JSON:\n{"keep_ids": ["A"]}\nDone.')
        self.assertEqual(parsed, {"keep_ids": ["A"]})

    def test_salvage_label_meta_selected_ids_when_comma_is_missing(self):
        raw = (
            '{"selected_candidate_ids": ["COM-C0004", "COM-C0012", "COM-C0013"]\n'
            '"rejected_candidate_ids": []}'
        )
        parsed = parse_json_object(raw, schema=LABEL_META_SCHEMA)
        self.assertEqual(parsed["selected_candidate_ids"], ["COM-C0004", "COM-C0012", "COM-C0013"])
        self.assertEqual(parsed["expansion_requests"], [])
        self.assertTrue(parsed["_salvaged_from_malformed_json"])

    def test_label_meta_schema_bounds_candidate_and_sentence_ids(self):
        schema = label_meta_schema(
            ["COM-C0001", "COM-C0002"],
            ["S0002", "S0003"],
            "ComparisonGroup",
        )
        selected = schema["properties"]["selected_candidate_ids"]
        self.assertEqual(selected["maxItems"], 2)
        self.assertEqual(selected["items"]["enum"], ["COM-C0001", "COM-C0002"])
        request = schema["properties"]["expansion_requests"]
        self.assertEqual(request["maxItems"], 5)
        self.assertEqual(
            request["items"]["properties"]["sentence_id"]["enum"],
            ["S0002", "S0003"],
        )
        proof = schema["properties"]["consensus_proof"]
        self.assertEqual(proof["items"]["properties"]["candidate_id"]["enum"], ["COM-C0001", "COM-C0002"])
        self.assertEqual(proof["items"]["properties"]["evidence_sentence_id"]["enum"], ["S0002", "S0003"])
        source_search = schema["properties"]["source_search_requests"]
        self.assertEqual(source_search["items"]["properties"]["label"]["enum"], ["ComparisonGroup"])

    def test_doctor_schema_requires_one_decision_per_visible_candidate(self):
        schema = doctor_review_schema(
            ["OUT-C0001", "OUT-C0002"],
            "Outcome",
        )
        decisions = schema["properties"]["decisions"]
        self.assertEqual(decisions["minItems"], 2)
        self.assertEqual(decisions["maxItems"], 2)
        self.assertEqual(
            decisions["items"]["properties"]["candidate_id"]["enum"],
            ["OUT-C0001", "OUT-C0002"],
        )

    def test_completion_budget_is_capped_by_remaining_context(self):
        budget = completion_token_budget(
            system="system",
            user="x" * 50000,
            configured_max=4096,
            context_window=16384,
            reserve_tokens=128,
            minimum_output_tokens=256,
        )
        self.assertLess(budget, 4096)
        self.assertGreaterEqual(budget, 256)

    def test_context_limit_error_parsing(self):
        message = (
            "'max_tokens' is too large: 4096. This model's maximum context "
            "length is 16384 tokens and your request has 12357 input tokens "
            "(4096 > 16384 - 12357)."
        )
        self.assertEqual(context_limits_from_error(message), (16384, 12357))

    def test_salvage_doctor_decisions_when_justification_has_unescaped_quote(self):
        schema = doctor_review_schema(
            ["INT-C0008", "INT-C0009"],
            "Intervention",
        )
        raw = """
        {
          "decisions": [
            {
              "candidate_id": "INT-C0008",
              "decision": "keep",
              "issue_type": "acceptable",
              "justification": "Describes an intervention."
            },
            {
              "candidate_id": "INT-C0009",
              "decision": "reject",
              "issue_type": "incomplete_phrase",
              "justification": "The text "Alternative?" is incomplete."
            }
          ],
          "expansion_requests": [],
          "missing_label": false,
          "comment": ""
        }
        """
        parsed = parse_json_object(raw, schema=schema)
        decisions = {
            item["candidate_id"]: item["decision"]
            for item in parsed["decisions"]
        }
        self.assertEqual(
            decisions,
            {"INT-C0008": "keep", "INT-C0009": "reject"},
        )
        self.assertTrue(parsed["_discarded_malformed_justifications"])

    def test_partial_doctor_salvage_does_not_invent_missing_decision(self):
        schema = doctor_review_schema(
            ["INT-C0008", "INT-C0009"],
            "Intervention",
        )
        raw = (
            '{"decisions": [{"candidate_id": "INT-C0008", '
            '"decision": "keep", "issue_type": "acceptable", '
            '"justification": "broken "quote"}]}'
        )
        parsed = parse_json_object(raw, schema=schema)
        self.assertEqual(
            [item["candidate_id"] for item in parsed["decisions"]],
            ["INT-C0008"],
        )


if __name__ == "__main__":
    unittest.main()
