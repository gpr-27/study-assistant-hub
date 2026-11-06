"""The SPA shell + assets ship with fixed filenames (no content hashing), so LOCAL dev serves
them with ``Cache-Control: no-cache`` — otherwise browsers cache them heuristically and an edit
never appears without a manual hard refresh. On Vercel the CDN serves a build whose asset URLs are
content-stamped (``?v=``), so vercel.json can cache those immutably; those rules are tested below."""
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


class StaticCacheHeaderTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_index_is_no_cache(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")

    def test_assets_are_no_cache(self):
        # app.js is always shipped in the SPA bundle.
        resp = self.client.get("/assets/app.js")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")


ROOT = Path(__file__).resolve().parent.parent
VERCEL = json.loads((ROOT / "vercel.json").read_text())


def _header_rules(source):
    return [h for h in VERCEL["headers"] if h["source"] == source]


class VercelConfigTests(unittest.TestCase):
    def test_api_is_rewritten_to_the_single_python_function(self):
        rw = {r["source"]: r["destination"] for r in VERCEL["rewrites"]}
        self.assertEqual(rw.get("/api/:path*"), "/api/index")
        self.assertTrue((ROOT / "api" / "index.py").is_file())

    def test_function_limits_and_region(self):
        fn = VERCEL["functions"]["api/index.py"]
        self.assertEqual(fn["maxDuration"], 60)
        self.assertEqual(VERCEL["regions"], ["bom1"])  # Atlas cluster is AWS ap-south-1
        self.assertIn("frontend/**", fn["excludeFiles"])
        for dropped in ("tests/**", "legacy/**", "interview/**", "GPR/**", "faiss_index/**", ".venv/**", "*.md"):
            self.assertIn(dropped, fn["excludeFiles"])
        for kept in ("data/index/**", "documents/**", "scripts/**"):  # shipped read-only / importable
            self.assertNotIn(kept, fn["excludeFiles"])

    def test_stamped_assets_are_immutable_and_unstamped_are_revalidated(self):
        rules = _header_rules("/assets/:path*")
        immutable = [r for r in rules if "has" in r][0]
        revalidate = [r for r in rules if "missing" in r][0]
        self.assertEqual(immutable["has"][0]["key"], "v")
        self.assertIn("immutable", immutable["headers"][0]["value"])
        self.assertEqual(revalidate["missing"][0]["key"], "v")
        self.assertIn("must-revalidate", revalidate["headers"][0]["value"])
        # The shell must never be cached long (it carries the version stamps).
        self.assertIn("max-age=0", _header_rules("/")[0]["headers"][0]["value"])

    def test_shell_headers_and_csp_cover_clerk_and_cdns(self):
        for src in ("/", "/index.html"):
            self.assertIn("max-age=0", _header_rules(src)[0]["headers"][0]["value"])
        shell = [h for h in VERCEL["headers"] if h["source"].startswith("/((?!api/)")][0]
        csp = {h["key"]: h["value"] for h in shell["headers"]}["Content-Security-Policy"]
        for host in ("cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com", "fonts.gstatic.com",
                     "*.clerk.accounts.dev", "*.clerk.com", "clerk.com", "challenges.cloudflare.com"):
            self.assertIn(host, csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("object-src 'none'", csp)

    def test_runtime_config_js_is_gone(self):
        self.assertFalse(any(h["source"] == "/runtime-config.js" for h in VERCEL["headers"]))
        self.assertEqual(TestClient(app).get("/runtime-config.js").status_code, 404)

    def test_cron_points_at_a_protected_route(self):
        crons = {c["path"]: c["schedule"] for c in VERCEL["crons"]}
        self.assertIn("/api/cron/janitor", crons)
        self.assertEqual(len(crons["/api/cron/janitor"].split()), 5)

    def test_config_has_no_secrets(self):
        text = (ROOT / "vercel.json").read_text()
        self.assertFalse(re.search(r"(sk-|gsk_|AIza|mongodb(\+srv)?://|CLERK_SECRET)", text))


class RuntimeRequirementsTests(unittest.TestCase):
    FORBIDDEN = ("langchain", "faiss", "tiktoken", "groq", "gunicorn", "uvicorn", "pytest", "anthropic")

    def test_runtime_requirements_are_trimmed(self):
        names = [re.split(r"[<>=\[ ;#]", ln.strip(), maxsplit=1)[0].lower()
                 for ln in (ROOT / "requirements.txt").read_text().splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        for bad in self.FORBIDDEN:
            self.assertFalse(any(n.startswith(bad) for n in names), f"{bad} must not be a runtime dependency")
        for needed in ("fastapi", "pydantic", "httpx", "pypdf", "numpy", "pymongo", "pyjwt", "python-multipart"):
            self.assertIn(needed, names)

    def test_dev_requirements_carry_server_and_tests(self):
        dev = (ROOT / "requirements-dev.txt").read_text()
        self.assertIn("uvicorn", dev)
        self.assertIn("pytest", dev)


class BuildFrontendTests(unittest.TestCase):
    def test_build_stamps_assets_without_touching_the_source(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_frontend", ROOT / "scripts" / "build_frontend.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        before = (ROOT / "frontend" / "index.html").read_bytes()
        out = Path(tempfile.mkdtemp()) / "dist"
        self.addCleanup(shutil.rmtree, out.parent, ignore_errors=True)
        mod.build(out)
        html = (out / "index.html").read_text()
        self.assertRegex(html, r'/assets/app\.js\?v=[0-9a-f]{10}')
        self.assertTrue((out / "assets" / "app.js").is_file())
        self.assertEqual((ROOT / "frontend" / "index.html").read_bytes(), before)  # source untouched


if __name__ == "__main__":
    unittest.main()
