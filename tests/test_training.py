"""Dataset checks work without PyTorch, model downloads, or a GPU."""

import json
from pathlib import Path
import tempfile
import unittest

from training.train_report import HERE, NOT_FOUND, load_examples, validate_splits


class ReportDatasetTests(unittest.TestCase):
    def load_row(self, row):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            return load_examples(path)

    def test_supplied_examples_and_separate_evaluation(self):
        train = load_examples(HERE / "data" / "report_train.jsonl")
        evaluation = load_examples(HERE / "data" / "report_eval.jsonl")
        self.assertEqual((len(train), len(evaluation)), (12, 4))
        validate_splits(train, evaluation)
        self.assertTrue(any(row["messages"][-1]["content"] == NOT_FOUND for row in train))
        self.assertTrue(any(row["messages"][-1]["content"] == NOT_FOUND for row in evaluation))

    def test_rejects_invalid_role_and_report_order(self):
        row = load_examples(HERE / "data" / "report_train.jsonl")[0]
        row["messages"][-1]["role"] = "user"
        with self.assertRaisesRegex(ValueError, "assistant message"):
            self.load_row(row)
        row["messages"][-1]["role"] = "assistant"
        row["messages"][-1]["content"] = "## Findings\nFacts [1]\n## Summary\nSummary\n## Next steps\nReview"
        with self.assertRaisesRegex(ValueError, "in order"):
            self.load_row(row)

    def test_rejects_missing_citations(self):
        row = load_examples(HERE / "data" / "report_train.jsonl")[0]
        row["messages"][-1]["content"] = row["messages"][-1]["content"].replace("[1]", "")
        with self.assertRaisesRegex(ValueError, "source citation"):
            self.load_row(row)

    def test_rejects_duplicate_and_leaked_prompts(self):
        rows = load_examples(HERE / "data" / "report_train.jsonl")
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits(rows, rows[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_splits([rows[0], rows[0]], rows[1:2])

    def test_rejects_empty_files_and_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no examples"):
                load_examples(path)
            path.write_text("{invalid}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()
