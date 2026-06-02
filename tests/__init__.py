"""Test suite for the Study Assistant Hub backend (stdlib unittest).

Seed the configuration the app validates at startup BEFORE any backend module is
imported, so the test suite is hermetic (it does not depend on a developer's local
.env). ``setdefault`` is used so a real .env / real environment still wins, and
``load_dotenv`` (called in backend.config) does not override already-set values.
This runs first because importing any ``tests.test_*`` module imports this package.
"""
import os

_TEST_DEFAULTS = {
    "APP_ENV": "development",
    "HOST": "127.0.0.1",
    "PORT": "8000",
    "LOG_LEVEL": "WARNING",
    "LLM_PROVIDER": "groq",
    "GROQ_API_KEY": "test-key-not-used-mocks-only",
    "AVAILABLE_MODELS": "test-model-a,test-model-b",
    "DEFAULT_MODEL": "test-model-a",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    # Keep auth + persistence OFF in tests regardless of a developer's local .env,
    # so the suite never reaches out to Clerk or MongoDB.
    "CLERK_PUBLISHABLE_KEY": "",
    "CLERK_SECRET_KEY": "",
    "MONGODB_URI": "",
}
for _key, _value in _TEST_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
