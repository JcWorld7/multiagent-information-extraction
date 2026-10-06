"""OpenAI-compatible vLLM client with structured output support."""
from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from Labelcentered.settings import PipelineSettings


class LLMClient:
    def __init__(self, settings: PipelineSettings, mock_responses: Optional[List[Dict[str, Any]]] = None):
        self.settings = settings
        self.mock_responses = list(mock_responses or [])
        self._client = None

    def _openai_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.llm_timeout_seconds,
            )
        return self._client

    def available_models(self) -> List[str]:
        response = self._openai_client().models.list()
        return [item.id for item in response.data]

    def validate_model_available(self) -> None:
        try:
            model_ids = self.available_models()
        except Exception as exc:
            raise RuntimeError(
                f"Could not query vLLM models at {self.settings.llm_base_url}. "
                "Start the selected backend server first, or pass --skip_llm_model_check "
                "if the server intentionally hides /v1/models."
            ) from exc
        if self.settings.llm_model in model_ids:
            return
        served = ", ".join(model_ids) if model_ids else "(no models reported)"
        raise RuntimeError(
            f"LLM backend {self.settings.llm_backend!r} expects model "
            f"{self.settings.llm_model!r}, but {self.settings.llm_base_url} "
            f"is serving: {served}. Start vLLM for this backend with "
            f"scripts/start_vllm_backend.sh {self.settings.llm_backend}, or pass "
            "--llm_model with the exact served model name/alias."
        )

    def structured_chat(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        call_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mock_responses:
            return self.mock_responses.pop(0)
        active_model = model or self.settings.llm_model
        source_name = f"{call_name or 'structured call'} using {active_model}"
        raw = self._chat_completion(active_model, system, user, schema, source_name)
        try:
            return parse_json_object(raw, source=f"LLM structured output from {source_name}", schema=schema)
        except ValueError as first_error:
            repair_system = (
                "You repair JSON. Return one valid JSON object matching the schema. "
                "Do not include markdown, comments, or explanatory text. Candidate-ID "
                "arrays must contain only candidate IDs allowed by the schema. Never "
                "enumerate source-unit or sentence IDs unless a single sentence_id "
                "field explicitly requires one."
            )
            repair_user = (
                "The previous model response was not valid JSON. Repair it without "
                "changing IDs, decisions, labels, or requested operations.\n\n"
                f"INVALID_RESPONSE:\n{raw[:12000]}"
            )
            repaired = self._chat_completion(active_model, repair_system, repair_user, schema, source_name)
            try:
                return parse_json_object(repaired, source=f"LLM repaired JSON from {source_name}", schema=schema)
            except ValueError as second_error:
                fallback = self._unstructured_json_completion(
                    active_model,
                    system,
                    user,
                    schema,
                    source_name,
                )
                try:
                    parsed = parse_json_object(
                        fallback,
                        source=f"LLM unstructured JSON fallback from {source_name}",
                        schema=schema,
                    )
                    parsed["_used_unstructured_json_fallback"] = True
                    return parsed
                except ValueError as third_error:
                    raise ValueError(
                        f"{first_error}\nRepair attempt also failed: {second_error}\n"
                        f"Unstructured JSON fallback also failed: {third_error}"
                    ) from third_error

    def _chat_completion(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        source_name: str,
        use_structured_outputs: bool = True,
    ) -> str:
        try:
            max_tokens = completion_token_budget(
                system=system,
                user=user,
                configured_max=self.settings.llm_max_tokens,
                context_window=self.settings.llm_context_window,
                reserve_tokens=self.settings.llm_context_reserve_tokens,
                minimum_output_tokens=self.settings.llm_min_output_tokens,
            )
        except ValueError as exc:
            raise ValueError(f"{source_name}: {exc}") from exc
        try:
            resp = self._create_completion(model, system, user, schema, max_tokens, use_structured_outputs)
        except Exception as exc:
            limits = context_limits_from_error(str(exc))
            if not limits:
                raise
            context_window, input_tokens = limits
            available = context_window - input_tokens - self.settings.llm_context_reserve_tokens
            retry_tokens = min(self.settings.llm_max_tokens, available)
            if retry_tokens < self.settings.llm_min_output_tokens:
                raise ValueError(
                    f"{source_name}: prompt uses {input_tokens} of "
                    f"{context_window} context tokens, "
                    f"leaving only {available} after reserve. Reduce doctor batch size "
                    "or source context before retrying."
                ) from exc
            if retry_tokens >= max_tokens:
                raise
            resp = self._create_completion(model, system, user, schema, retry_tokens, use_structured_outputs)
        raw = resp.choices[0].message.content
        if raw is None:
            raise ValueError(f"LLM returned empty content for model {model}")
        return raw

    def _unstructured_json_completion(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        source_name: str,
    ) -> str:
        compact_schema = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
        fallback_system = (
            f"{system}\n\n"
            "Return one valid JSON object and nothing else. Do not use markdown. "
            "The JSON object must match the schema exactly. Candidate-ID arrays "
            "must contain only IDs allowed by the schema."
        )
        fallback_user = (
            f"{user}\n\nJSON_SCHEMA:\n{compact_schema}\n\n"
            "Return only the JSON object."
        )
        return self._chat_completion(
            model,
            fallback_system,
            fallback_user,
            schema,
            source_name,
            use_structured_outputs=False,
        )

    def _create_completion(
        self,
        model: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_tokens: int,
        use_structured_outputs: bool = True,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": self.settings.llm_temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if use_structured_outputs:
            kwargs["extra_body"] = {"structured_outputs": {"json": schema}}
        return self._openai_client().chat.completions.create(**kwargs)


def parse_json_object(raw: Any, source: str = "JSON", schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"{source} was {type(raw).__name__}, expected string or object")

    text = strip_code_fence(raw.strip())
    attempts = [text]
    balanced = first_balanced_json_object(text)
    if balanced and balanced != text:
        attempts.append(balanced)
    without_trailing_commas = remove_trailing_commas(balanced or text)
    if without_trailing_commas not in attempts:
        attempts.append(without_trailing_commas)

    last_error: JSONDecodeError | None = None
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            raise ValueError(f"{source} decoded to {type(parsed).__name__}, expected object")
        return parsed

    salvaged = salvage_doctor_review(text, schema)
    if salvaged is None:
        salvaged = salvage_schema_object(text, schema)
    if salvaged is not None:
        salvaged["_salvaged_from_malformed_json"] = True
        return salvaged

    if last_error is None:
        raise ValueError(f"{source} did not contain a JSON object. Preview: {text[:500]}")
    raise ValueError(
        f"Invalid {source}: {last_error}. "
        f"Nearby text: {json_error_context(attempts[-1], last_error.pos)}"
    )


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def salvage_doctor_review(text: str, schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Recover candidate decisions while discarding malformed free text."""
    if not schema or schema.get("type") != "object":
        return None
    decisions_schema = schema.get("properties", {}).get("decisions", {})
    item_properties = decisions_schema.get("items", {}).get("properties", {})
    candidate_schema = item_properties.get("candidate_id", {})
    decision_schema = item_properties.get("decision", {})
    allowed_ids = candidate_schema.get("enum")
    allowed_decisions = set(decision_schema.get("enum", []))
    if not allowed_ids or allowed_decisions != {"keep", "reject"}:
        return None

    markers = []
    for candidate_id in allowed_ids:
        match = re.search(
            rf'"candidate_id"\s*:\s*"{re.escape(candidate_id)}"',
            text,
        )
        if match:
            markers.append((match.start(), candidate_id))
    markers.sort()
    recovered = []
    allowed_issues = set(item_properties.get("issue_type", {}).get("enum", []))
    for index, (start, candidate_id) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        segment = text[start:end]
        decision_match = re.search(
            r'"decision"\s*:\s*"(keep|reject)"',
            segment,
        )
        if not decision_match:
            continue
        issue_match = re.search(
            r'"issue_type"\s*:\s*"([^"]+)"',
            segment,
        )
        issue_type = issue_match.group(1) if issue_match else "unsupported"
        if issue_type not in allowed_issues:
            issue_type = "unsupported"
        recovered.append(
            {
                "candidate_id": candidate_id,
                "decision": decision_match.group(1),
                "issue_type": issue_type,
                "justification": "Decision recovered from malformed JSON; original justification omitted.",
            }
        )
    if not recovered:
        return None
    missing_label_match = re.search(
        r'"missing_label"\s*:\s*(true|false)',
        text,
        flags=re.IGNORECASE,
    )
    return {
        "decisions": recovered,
        "expansion_requests": [],
        "missing_label": bool(
            missing_label_match
            and missing_label_match.group(1).lower() == "true"
        ),
        "comment": "Malformed DoctorAgent JSON salvaged deterministically.",
        "_discarded_malformed_justifications": True,
    }


def salvage_schema_object(text: str, schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Recover simple schema fields from malformed JSON without inventing IDs.

    This is intentionally narrow. It can recover complete scalar values and
    complete arrays after a named field, then fills missing schema-declared
    fields with safe defaults. Incomplete doctor reviews remain incomplete and
    are rejected by the review validator.
    """
    if not schema or schema.get("type") != "object":
        return None
    properties = schema.get("properties", {})
    recovered: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        value = salvage_field(text, field_name, field_schema)
        if value is not _MISSING:
            recovered[field_name] = value
    if not recovered:
        return None
    for field_name, field_schema in properties.items():
        if field_name not in recovered:
            recovered[field_name] = default_for_schema(field_schema)
    return recovered


_MISSING = object()


def salvage_field(text: str, field_name: str, field_schema: Dict[str, Any]) -> Any:
    field_type = field_schema.get("type")
    key_match = re.search(rf'"{re.escape(field_name)}"\s*:', text)
    if not key_match:
        return _MISSING
    value_start = key_match.end()
    remainder = text[value_start:].lstrip()
    if field_type == "array":
        return salvage_array(remainder)
    if field_type == "string":
        return salvage_string(remainder)
    if field_type == "boolean":
        return salvage_boolean(remainder)
    if field_type == "object":
        return salvage_object(remainder)
    return _MISSING


def salvage_array(text: str) -> Any:
    if not text.startswith("["):
        return _MISSING
    candidate = balanced_container(text, "[", "]")
    if not candidate:
        return _MISSING
    try:
        return json.loads(remove_trailing_commas(candidate))
    except JSONDecodeError:
        string_items = re.findall(r'"((?:\\.|[^"\\])*)"', candidate)
        if string_items:
            return [bytes(item, "utf-8").decode("unicode_escape") for item in string_items]
    return _MISSING


def salvage_object(text: str) -> Any:
    if not text.startswith("{"):
        return _MISSING
    candidate = balanced_container(text, "{", "}")
    if not candidate:
        return _MISSING
    try:
        return json.loads(remove_trailing_commas(candidate))
    except JSONDecodeError:
        return _MISSING


def salvage_string(text: str) -> Any:
    if not text.startswith('"'):
        return _MISSING
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except JSONDecodeError:
        return _MISSING
    return value if isinstance(value, str) else _MISSING


def salvage_boolean(text: str) -> Any:
    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False
    return _MISSING


def balanced_container(text: str, open_char: str, close_char: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return ""


def default_for_schema(field_schema: Dict[str, Any]) -> Any:
    field_type = field_schema.get("type")
    if field_type == "array":
        return []
    if field_type == "object":
        return {}
    if field_type == "boolean":
        return False
    if field_type == "string":
        return ""
    return None


def json_error_context(text: str, position: int, window: int = 240) -> str:
    start = max(0, position - window)
    end = min(len(text), position + window)
    return repr(text[start:end])


def rough_token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def completion_token_budget(
    *,
    system: str,
    user: str,
    configured_max: int,
    context_window: int,
    reserve_tokens: int,
    minimum_output_tokens: int,
) -> int:
    prompt_tokens = rough_token_estimate(system) + rough_token_estimate(user) + 32
    available = context_window - prompt_tokens - reserve_tokens
    if available < minimum_output_tokens:
        raise ValueError(
            f"Estimated prompt size is {prompt_tokens} tokens for a "
            f"{context_window}-token context window, leaving {available} output "
            "tokens. Reduce doctor batch size or candidate context."
        )
    return max(minimum_output_tokens, min(configured_max, available))


def context_limits_from_error(message: str) -> Optional[tuple[int, int]]:
    match = re.search(
        r"maximum context length is\s+(\d+)\s+tokens.*?"
        r"request has\s+(\d+)\s+input tokens",
        message,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))
