"""Data-independent regressions for the reorganized workflows (stdlib only)."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from finarg3_sm.paths import PROJECT_ROOT, required_input_path
from finarg3_sm.preprocessing import data_prep, features


def prompt_digest(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("SYSTEM_PROMPT", "FEWSHOT_META")
    }
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class StructureTests(unittest.TestCase):
    def test_project_root_is_independent_of_working_directory(self):
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parent.parent)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(data_prep.OUT_DIR, PROJECT_ROOT / "data")
            finally:
                os.chdir(previous)

    def test_input_configuration_never_guesses_missing_sources(self):
        for value in (None, "", "   "):
            env = {} if value is None else {"FINARG3_RAW_DIR": value}
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaisesRegex(SystemExit, "Set FINARG3_RAW_DIR"):
                    required_input_path("FINARG3_RAW_DIR")
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing")
            with patch.dict("os.environ", {"FINARG3_RAW_DIR": missing}):
                with self.assertRaisesRegex(SystemExit, "does not exist"):
                    required_input_path("FINARG3_RAW_DIR")

    def test_explicit_input_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"FINARG3_RAW_DIR": directory}):
                self.assertEqual(required_input_path("FINARG3_RAW_DIR"), Path(directory))

    def test_preparation_preserves_deduplication_folds_and_pair_labels(self):
        # Synthetic posts cover NFKC deduplication, invalid labels, short text,
        # stable ordering and a below-threshold pair. No organizer text is used.
        raw = [
            {"post_rationale": "ＡＢＣＤ　", "MPP": 0.10, "ML": 0.01},
            {"post_rationale": "ABCD", "MPP": 0.10, "ML": 0.02},
            {"post_rationale": "另一篇貼文", "MPP": 0.11, "ML": 0.03},
            {"post_rationale": "看空風險貼文", "MPP": -0.10, "ML": 0.04},
            {"post_rationale": "短", "MPP": 0.9, "ML": 0.0},
            {"post_rationale": "沒有有效標籤", "MPP": None, "ML": 0.0},
            {"post_rationale": "非數值標籤文", "MPP": float("nan"), "ML": 0.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Social_Media_Posts_with_MPP_ML.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with patch.dict("os.environ", {"FINARG3_RAW_DIR": directory}):
                posts = data_prep.assign_folds(data_prep.load_clean_posts())
        self.assertEqual(posts, [
            {"pid": 0, "text": "ABCD", "mpp": 0.10, "ml": 0.01, "fold": 1},
            {"pid": 1, "text": "另一篇貼文", "mpp": 0.11, "ml": 0.03, "fold": 2},
            {"pid": 2, "text": "看空風險貼文", "mpp": -0.10, "ml": 0.04, "fold": 0},
        ])
        self.assertEqual(data_prep.build_pairs(posts), [
            {"a_pid": 0, "b_pid": 2, "label": 1, "dmpp": 0.2, "a_fold": 1, "b_fold": 0},
            {"a_pid": 1, "b_pid": 2, "label": 1, "dmpp": 0.21, "a_fold": 2, "b_fold": 0},
        ])

    def test_features_preserve_ticker_and_stance_rules(self):
        result = features.post_features("2330 2026 看多買進 看空 EPS 5.5 營收年增20% https://example.org")
        self.assertEqual(result["ticker_count"], 1)  # year 2026 excluded
        self.assertEqual(result["stance_net"], 1)
        self.assertEqual(result["pct_count"], 1)
        self.assertEqual(result["url_count"], 1)
        self.assertEqual(features.post_features("")["log_len"], 0)

    def test_original_and_generated_judge_prompts_match_baseline(self):
        judge_dir = PROJECT_ROOT / "finarg3_sm" / "judging"
        self.assertEqual(prompt_digest(judge_dir / "llm_judge.py"),
                         "7dc3febd4e772f0e47f0bf62939ffad0c4c116a266beeaa5f7fa8f17def75dab")
        # Run the actual generator on a temporary copy, leaving source clean.
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for name in ("make_v3sa.py", "llm_judge.py"):
                (temporary / name).write_text((judge_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
            runpy.run_path(str(temporary / "make_v3sa.py"))
            self.assertEqual(prompt_digest(temporary / "llm_judge_v3sa.py"),
                             "0bed449469fedd0d4ec5052daead3a7779115b61494262a97bab7e705b39f171")


if __name__ == "__main__":
    unittest.main()
