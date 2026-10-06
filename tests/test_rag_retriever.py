import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.rag_retriever import GuidelineRetriever, LABELS


class GuidelineRetrieverTests(unittest.TestCase):
    def test_static_guidelines_include_codebook_domain_rules(self):
        rows = GuidelineRetriever(guideline_dir=None).retrieve("Participant")
        text = "\n".join(rows)
        self.assertIn("K-12 students", text)
        self.assertIn("pure counts", text)
        self.assertIn("smallest phrase", text)
        self.assertIn("827 upper middle school students", text)
        self.assertIn("copy only text present", text)

    def test_statistical_analysis_allows_broad_analytic_spans(self):
        rows = GuidelineRetriever(guideline_dir=None).retrieve("StatisticalAnalysis")
        text = "\n".join(rows)
        self.assertIn("Broad spans are allowed", text)
        self.assertIn("intention-to-treat", text)

    def test_label_txt_guideline_is_loaded_and_bounded(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "Outcome.txt").write_text("outcome note " * 300, encoding="utf-8")
            rows = GuidelineRetriever(directory).retrieve("Outcome")
        local = [row for row in rows if row.startswith("[local guideline:")]
        self.assertEqual(len(local), 1)
        self.assertLessEqual(len(local[0]), 1500)

    def test_unknown_label_returns_empty(self):
        self.assertEqual(GuidelineRetriever(guideline_dir=None).retrieve("NotALabel"), [])


if __name__ == "__main__":
    unittest.main()
