"""The build-time minifiers only drop comments/whitespace; output must still parse."""
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import minify_assets as m  # noqa: E402

NODE = shutil.which("node")


class CssTests(unittest.TestCase):
    def test_strips_comments_and_space_keeps_strings_and_calc(self):
        out = m.minify_css('/* c */ a  >  b , .x:hover {\n  color: red ;\n  width: calc(100% - 2px) ;\n  content: "a  /* no */  b";\n}\n')
        self.assertEqual(out, 'a>b,.x:hover{color:red;width:calc(100% - 2px);content:"a  /* no */  b"}')

    def test_keeps_descendant_pseudo_space(self):
        self.assertEqual(m.minify_css("a :hover { x: 1 }"), "a :hover{x:1}")

    def test_real_stylesheets_stay_balanced(self):
        for f in (ROOT / "frontend/assets").glob("*.css"):
            o = m.minify_css(f.read_text())
            self.assertEqual(o.count("{"), o.count("}"), f.name)
            self.assertLess(len(o), len(f.read_text()))


class JsTests(unittest.TestCase):
    def test_preserves_strings_templates_regex(self):
        src = 'const a = "x // y"; // gone\nconst r = /[/]\\/\\/ab/g; /* c */\nconst t = `line1\n    keep   ${ f(`in ${1}`) }  // not a comment\n`;\n'
        out = m.minify_js(src)
        self.assertIn('"x // y"', out)
        self.assertIn("/[/]\\/\\/ab/g", out)
        self.assertIn("line1\n    keep   ${", out)
        self.assertIn("// not a comment", out)
        self.assertNotIn("gone", out)

    @unittest.skipUnless(NODE, "node not installed")
    def test_every_frontend_script_still_parses(self):
        import tempfile
        for f in sorted((ROOT / "frontend/assets").rglob("*.js")):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / ("m.mjs" if "import(" in f.read_text() and f.name == "auth.js" else "m.js")
                out.write_text(m.minify_js(f.read_text()))
                r = subprocess.run([NODE, "--check", str(out)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, f"{f.name}: {r.stderr[:300]}")


if __name__ == "__main__":
    unittest.main()
