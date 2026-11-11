import os
from pathlib import Path
import re
import unittest

from src.version import __version__


ROOT = Path(__file__).resolve().parents[1]


class ReleaseVersionTests(unittest.TestCase):
    def test_project_version_matches_application(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
        self.assertIsNotNone(version)
        self.assertEqual(version.group(1), __version__)

    def test_lockfile_version_matches_application(self):
        lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
        version = re.search(r'name = "dianxin-auto-train"\nversion = "([^"]+)"', lock)
        self.assertIsNotNone(version)
        self.assertEqual(version.group(1), __version__)

    def test_changelog_has_unreleased_and_dated_release(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [Unreleased]", changelog)
        self.assertRegex(
            changelog,
            rf"(?m)^## \[{re.escape(__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
        )
        section = changelog.split(f"## [{__version__}] - ", 1)[1].split("\n## [", 1)[0]
        self.assertRegex(section, r"(?m)^### (Added|Changed|Deprecated|Removed|Fixed|Security)$")

    @unittest.skipUnless(os.getenv("RELEASE_TAG"), "Only checked for a release tag")
    def test_release_tag_matches_application(self):
        self.assertEqual(os.environ["RELEASE_TAG"], f"v{__version__}")


if __name__ == "__main__":
    unittest.main()
