import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.settings import (
    LLM_BACKENDS,
    PipelineSettings,
    llm_backend_metadata,
    llm_backend_output_dir,
    llm_backend_spec,
)


class BackendSettingsTests(unittest.TestCase):
    def test_configured_backend_models_and_output_dirs(self):
        self.assertEqual(
            llm_backend_spec("qwen").model,
            "Qwen/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(
            llm_backend_spec("llama31_8b").model,
            "meta-llama/Llama-3.1-8B-Instruct",
        )
        self.assertEqual(
            llm_backend_spec("mistral7b").model,
            "mistralai/Mistral-7B-Instruct-v0.3",
        )
        self.assertEqual(str(llm_backend_output_dir("qwen")), "outputs/qwen")
        self.assertEqual(str(llm_backend_output_dir("llama31_8b")), "outputs/llama31_8b")
        self.assertEqual(str(llm_backend_output_dir("mistral7b")), "outputs/mistral7b")

    def test_backend_metadata_uses_settings_without_tokens(self):
        settings = PipelineSettings(
            llm_backend="llama31_8b",
            llm_model=LLM_BACKENDS["llama31_8b"].model,
        )
        metadata = llm_backend_metadata(settings)
        self.assertEqual(metadata["backend"], "llama31_8b")
        self.assertEqual(metadata["model"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertNotIn("token", {key.lower() for key in metadata})

    def test_unknown_backend_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown LLM backend"):
            llm_backend_spec("not-a-backend")


if __name__ == "__main__":
    unittest.main()
