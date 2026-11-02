"""Synthetic checks of the public Git-content boundary; no patient fixtures."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "public_release_check", Path(__file__).resolve().parents[1] / "scripts/check_public_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class PublicReleaseTests(unittest.TestCase):
    def test_paths_are_explicit_and_data_are_local(self):
        for name in ("src/ai4ct/paths.py", "tests/test_workspace.py", "README.md",
                     "configs/public/model_configurations.json"):
            self.assertTrue(release.allowed_path(name))
        for name in ("data/raw/labels.json", "outputs/weights.pt", "docs/data_inventory.md",
                     "configs/quantum_benchmark/execution_v1.json", "src/ai4ct/patient.csv",
                     "src/ai4ct/../../data/raw/record.py", "src/ai4ct/model.joblib"):
            self.assertFalse(release.allowed_path(name))

    def test_private_config_fields_rejected_recursively(self):
        self.assertTrue(release.private_keys({"cohorts": [{"patient_id": "synthetic"}]}))
        self.assertTrue(release.private_keys({"manifest_sha256": "synthetic"}))
        self.assertFalse(release.private_keys({"model_configs": {"demo": {"epochs": 3}}}))

    def run_fake_tree(self, listing, content):
        def fake_git(*args):
            return content if args[0] == "cat-file" else listing
        with patch.object(release, "git", side_effect=fake_git), contextlib.redirect_stdout(io.StringIO()) as stream:
            code = release.audit(staged=True)
        return code, stream.getvalue()

    def test_text_source_passes_but_binary_and_symlink_fail(self):
        normal = b"100644 abc 0\tsrc/ai4ct/example.py\0"
        self.assertEqual(self.run_fake_tree(normal, b"print('synthetic')\n")[0], 0)
        self.assertEqual(self.run_fake_tree(normal, b"binary\0data")[0], 1)
        self.assertEqual(self.run_fake_tree(b"120000 abc 0\tsrc/ai4ct/example.py\0", b"target")[0], 1)

    def test_secret_is_rejected_without_echoing_value(self):
        token = "ghp_" + "A" * 40
        code, output = self.run_fake_tree(b"100644 abc 0\tsrc/ai4ct/example.py\0", token.encode())
        self.assertEqual(code, 1)
        self.assertNotIn(token, output)
        self.assertIn("github_token", output)

    def test_empty_tree_is_not_a_success(self):
        self.assertEqual(self.run_fake_tree(b"", b"")[0], 1)


if __name__ == "__main__":
    unittest.main()
