"""Unit tests for Clerk session-JWT verification in ``backend.auth`` (no network).

A locally generated RSA key stands in for Clerk's signing key; the JWKS client is replaced with
a stub that serves it, so signature/claim handling, the token cache, JWKS caching/rotation
throttling, outage behaviour and owner resolution are all exercised hermetically. (The live
equivalent against a real Clerk dev instance is ``scripts/clerk_e2e.py``.)
"""
import base64
import time
import unittest
from unittest import mock

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from backend import auth, config

HOST = "clerk.example.test"
PK = "pk_test_" + base64.b64encode(f"{HOST}$".encode()).decode().rstrip("=")
ISS = f"https://{HOST}"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk(key, kid):
    d = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    d.update(kid=kid, use="sig", alg="RS256")
    return jwt.PyJWK.from_dict(d)


class StubClient:
    """Mimics the bits of PyJWKClient that auth.py uses."""

    def __init__(self, keys, rotated=None):
        self.keys = keys
        self.rotated = rotated  # served after refresh=True (simulates Clerk key rotation)
        self.cached_calls = 0
        self.refresh_calls = 0

    def get_signing_keys(self, refresh=False):
        if refresh:
            self.refresh_calls += 1
            if self.rotated is not None:
                self.keys = self.rotated
        else:
            self.cached_calls += 1
        return self.keys


def make_token(key=_KEY, kid="ins_kid1", **overrides):
    now = int(time.time())
    claims = {"sub": "user_abc", "iss": ISS, "iat": now, "nbf": now, "exp": now + 60, "sts": "active", "sid": "sess_1"}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, "RS256", headers={"kid": kid})


class ClerkVerifyBase(unittest.TestCase):
    def setUp(self):
        auth._token_cache.clear()
        auth._jwks_failed_until = 0.0
        auth._jwks_last_forced_refresh = None
        self.stub = StubClient([_jwk(_KEY, "ins_kid1")])
        patches = [
            mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", PK),
            mock.patch.object(config, "AUTH_ENABLED", True),
            mock.patch.object(auth, "_jwks_client", lambda: self.stub),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(auth._token_cache.clear)
        self.addCleanup(setattr, auth, "_jwks_failed_until", 0.0)
        self.addCleanup(setattr, auth, "_jwks_last_forced_refresh", None)


class VerifyClaimsTests(ClerkVerifyBase):
    def test_issuer_derived_from_publishable_key(self):
        self.assertEqual(auth._frontend_api_from_pk(PK), HOST)
        self.assertEqual(auth._issuer(), ISS)
        self.assertIsNone(auth._frontend_api_from_pk("garbage"))

    def test_valid_token_returns_sub(self):
        self.assertEqual(auth._verify_clerk_token(make_token()), "user_abc")

    def test_expired_token_rejected(self):
        past = int(time.time()) - 120
        self.assertIsNone(auth._verify_clerk_token(make_token(iat=past - 60, nbf=past - 60, exp=past)))

    def test_small_clock_skew_tolerated(self):
        self.assertEqual(auth._verify_clerk_token(make_token(exp=int(time.time()) - 5)), "user_abc")

    def test_not_yet_valid_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(make_token(nbf=int(time.time()) + 600)))

    def test_wrong_issuer_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(make_token(iss="https://evil.example.com")))

    def test_missing_issuer_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(make_token(iss=None)))

    def test_missing_sub_or_exp_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(make_token(sub=None)))
        self.assertIsNone(auth._verify_clerk_token(make_token(exp=None)))

    def test_non_string_or_empty_sub_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(make_token(sub=12345)))
        self.assertIsNone(auth._verify_clerk_token(make_token(sub="")))

    def test_pending_session_rejected_and_not_cached(self):
        tok = make_token(sts="pending")
        self.assertIsNone(auth._verify_clerk_token(tok))
        self.assertEqual(auth._token_cache, {})

    def test_wrong_signing_key_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(make_token(key=_OTHER_KEY)))

    def test_tampered_payload_rejected(self):
        h, p, s = make_token().split(".")
        forged = base64.urlsafe_b64encode(b'{"sub":"user_victim","iss":"%s","exp":9999999999}' % ISS.encode()).rstrip(b"=").decode()
        self.assertIsNone(auth._verify_clerk_token(f"{h}.{forged}.{s}"))

    def test_alg_none_and_hs256_rejected(self):
        self.assertIsNone(auth._verify_clerk_token(jwt.encode({"sub": "user_abc", "iss": ISS, "exp": int(time.time()) + 60},
                                                              None, "none", headers={"kid": "ins_kid1"})))
        pub_pem = _KEY.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        try:
            hs = jwt.encode({"sub": "user_abc", "iss": ISS, "exp": int(time.time()) + 60}, pub_pem, "HS256",
                            headers={"kid": "ins_kid1"})
        except Exception:  # newer PyJWT refuses to even sign with a PEM key: that is the point
            return
        self.assertIsNone(auth._verify_clerk_token(hs))

    def test_garbage_and_missing_kid_rejected(self):
        self.assertIsNone(auth._verify_clerk_token("not.a.jwt"))
        self.assertIsNone(auth._verify_clerk_token("x"))
        nokid = jwt.encode({"sub": "user_abc", "iss": ISS, "exp": int(time.time()) + 60}, _KEY, "RS256")
        self.assertIsNone(auth._verify_clerk_token(nokid))

    def test_azp_and_audience_not_required(self):
        # Real Clerk session tokens carry neither; a stray aud must not break verification.
        self.assertEqual(auth._verify_clerk_token(make_token(aud="something")), "user_abc")


class TokenCacheTests(ClerkVerifyBase):
    def test_repeat_token_skips_jwks_and_verify(self):
        tok = make_token()
        self.assertEqual(auth._verify_clerk_token(tok), "user_abc")
        calls = self.stub.cached_calls
        for _ in range(5):
            self.assertEqual(auth._verify_clerk_token(tok), "user_abc")
        self.assertEqual(self.stub.cached_calls, calls)

    def test_cache_ttl_capped_at_30s_and_never_past_exp(self):
        tok = make_token(exp=int(time.time()) + 3600)
        auth._verify_clerk_token(tok)
        (exp_m, _), = auth._token_cache.values()
        self.assertLessEqual(exp_m - time.monotonic(), 30.0)
        auth._token_cache.clear()
        auth._verify_clerk_token(make_token(exp=int(time.time()) + 1))   # <= 1 s left: not cached
        self.assertEqual(auth._token_cache, {})

    def test_cache_expiry_forces_reverify(self):
        tok = make_token()
        auth._verify_clerk_token(tok)
        key = next(iter(auth._token_cache))
        auth._token_cache[key] = (time.monotonic() - 1, "user_abc")
        before = self.stub.cached_calls
        self.assertEqual(auth._verify_clerk_token(tok), "user_abc")
        self.assertEqual(self.stub.cached_calls, before + 1)

    def test_cache_is_bounded(self):
        with mock.patch.object(auth, "_TOKEN_CACHE_MAX", 3):
            for i in range(10):
                auth._verify_clerk_token(make_token(sub=f"user_{i}"))
            self.assertLessEqual(len(auth._token_cache), 3)


class JwksBehaviourTests(ClerkVerifyBase):
    def test_unknown_kid_refetches_at_most_once_per_cooldown(self):
        for i in range(10):
            self.assertIsNone(auth._verify_clerk_token(make_token(kid=f"ins_junk{i}")))
        self.assertEqual(self.stub.refresh_calls, 1)

    def test_key_rotation_is_picked_up_via_refresh(self):
        new = _jwk(_OTHER_KEY, "ins_kid2")
        self.stub.rotated = [new]
        self.assertEqual(auth._verify_clerk_token(make_token(key=_OTHER_KEY, kid="ins_kid2")), "user_abc")
        self.assertEqual(self.stub.refresh_calls, 1)

    def test_outage_raises_unavailable_then_fails_fast(self):
        class Down(StubClient):
            def get_signing_keys(self, refresh=False):
                self.cached_calls += 1
                raise jwt.PyJWKClientConnectionError("down")

        self.stub = Down([])
        with self.assertRaises(auth.ClerkUnavailable):
            auth._verify_clerk_token(make_token())
        self.assertGreater(auth._jwks_failed_until, time.monotonic())
        calls = self.stub.cached_calls
        with self.assertRaises(auth.ClerkUnavailable):
            auth._verify_clerk_token(make_token())
        self.assertEqual(self.stub.cached_calls, calls)  # backoff: no further network attempt

    def test_cached_token_still_valid_during_outage(self):
        tok = make_token()
        self.assertEqual(auth._verify_clerk_token(tok), "user_abc")
        auth._jwks_failed_until = time.monotonic() + 30
        self.assertEqual(auth._verify_clerk_token(tok), "user_abc")

    def test_no_publishable_key_means_no_client(self):
        with mock.patch.object(auth, "_jwks_client", lambda: None):
            self.assertIsNone(auth._verify_clerk_token(make_token()))


class OwnerResolutionTests(ClerkVerifyBase):
    def setUp(self):
        super().setUp()
        app = FastAPI()

        @app.get("/who")
        def who(owner: str = Depends(auth.current_owner)):
            return {"owner": owner}

        @app.get("/clerk-only")
        def clerk_only(owner: str = Depends(auth.require_clerk_user)):
            return {"owner": owner}

        self.client = TestClient(app)
        self.touch = mock.patch.object(auth.db, "touch_guest", side_effect=lambda g: g == "guest_" + "a" * 32)
        self.touch_mock = self.touch.start()
        self.addCleanup(self.touch.stop)

    @staticmethod
    def bearer(tok):
        return {"Authorization": f"Bearer {tok}"}

    def test_valid_token_resolves_to_clerk_user(self):
        r = self.client.get("/who", headers=self.bearer(make_token()))
        self.assertEqual((r.status_code, r.json()), (200, {"owner": "user_abc"}))

    def test_missing_token_is_401(self):
        self.assertEqual(self.client.get("/who").status_code, 401)
        self.assertEqual(self.client.get("/clerk-only").status_code, 401)

    def test_bad_tokens_are_401_never_500(self):
        for tok in ("garbage", make_token(key=_OTHER_KEY), make_token(iss="https://evil.test"),
                    make_token(exp=int(time.time()) - 300), make_token(sts="pending")):
            self.assertEqual(self.client.get("/who", headers=self.bearer(tok)).status_code, 401, tok[:20])

    def test_bearer_scheme_is_case_insensitive_and_non_bearer_ignored(self):
        self.assertEqual(self.client.get("/who", headers={"Authorization": "bearer " + make_token()}).status_code, 200)
        self.assertEqual(self.client.get("/who", headers={"Authorization": "Basic abc"}).status_code, 401)

    def test_guest_header_cannot_override_valid_clerk_identity(self):
        guest = "guest_" + "a" * 32
        r = self.client.get("/who", headers={**self.bearer(make_token()), "X-Guest-Id": guest})
        self.assertEqual(r.json(), {"owner": "user_abc"})
        self.touch_mock.assert_not_called()

    def test_live_guest_without_token_is_the_guest(self):
        r = self.client.get("/who", headers={"X-Guest-Id": "guest_" + "a" * 32})
        self.assertEqual(r.json(), {"owner": "guest_" + "a" * 32})

    def test_forged_guest_ids_are_401(self):
        for gid in ("user_abc", "guest_" + "b" * 32, "local", "x" * 200):
            self.assertEqual(self.client.get("/who", headers={"X-Guest-Id": gid}).status_code, 401, gid[:12])

    def test_bad_token_with_live_guest_falls_back_to_guest_not_clerk(self):
        r = self.client.get("/who", headers={**self.bearer(make_token(key=_OTHER_KEY)), "X-Guest-Id": "guest_" + "a" * 32})
        self.assertEqual(r.json(), {"owner": "guest_" + "a" * 32})

    def test_guest_cannot_use_clerk_only_routes(self):
        r = self.client.get("/clerk-only", headers={"X-Guest-Id": "guest_" + "a" * 32})
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/clerk-only", headers=self.bearer(make_token()))
        self.assertEqual(r.json(), {"owner": "user_abc"})

    def test_jwks_outage_gives_retryable_503_not_401_or_guest(self):
        class Down(StubClient):
            def get_signing_keys(self, refresh=False):
                raise jwt.PyJWKClientConnectionError("down")

        self.stub = Down([])
        r = self.client.get("/who", headers={**self.bearer(make_token()), "X-Guest-Id": "guest_" + "a" * 32})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get("retry-after"), "5")
        # ...and the same outage still serves a pure guest / anonymous request correctly.
        self.assertEqual(self.client.get("/who", headers={"X-Guest-Id": "guest_" + "a" * 32}).status_code, 200)
        self.assertEqual(self.client.get("/who").status_code, 401)

    def test_auth_disabled_everyone_is_local(self):
        with mock.patch.object(config, "AUTH_ENABLED", False):
            self.assertEqual(self.client.get("/who").json(), {"owner": "local"})
            # a token is ignored entirely when Clerk is off
            self.assertEqual(self.client.get("/who", headers=self.bearer("junk")).json(), {"owner": "local"})


if __name__ == "__main__":
    unittest.main()
