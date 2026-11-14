"""The prepared checkout must resolve its own CLIP from any working directory."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from brtp.reproduce import copy_core


class DependencyLayoutTests(unittest.TestCase):
    def test_prepared_core_imports_its_bundled_clip_in_a_clean_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkout = root / "BRTP"
            copy_core(checkout)
            script = "\n".join(
                (
                    "import json, sys",
                    "from pathlib import Path",
                    "sys.path.insert(0, sys.argv[1])",
                    "import brtp",
                    "from clip import clip",
                    "from brtp.models.residual_transport import ResidualTransportCLIP",
                    "print(json.dumps({'version': brtp.__version__, 'clip': str(Path(clip.__file__).resolve())}))",
                )
            )
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script, str(checkout)],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=60,
            )
            loaded = json.loads(result.stdout)
            self.assertEqual(loaded["version"], "1.01")
            self.assertEqual(Path(loaded["clip"]), checkout / "CLIP/clip/clip.py")
            self.assertFalse((checkout / "vendor").exists())
