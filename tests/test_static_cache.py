"""The SPA shell + assets ship with fixed filenames (no content hashing), so they
must be served with ``Cache-Control: no-cache`` — otherwise browsers cache them
heuristically and a deploy never appears without a manual hard refresh."""
import unittest

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


if __name__ == "__main__":
    unittest.main()
