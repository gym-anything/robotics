"""Exercise the real titration selector over HTTP, with no physics process."""
import unittest
import urllib.error

from tests import test_http_server as helpers
from titrate.newton15.variations import contract, select


class TitrationCatalogHTTPTest(unittest.TestCase):
    setUp = helpers.HTTPServerTest.setUp
    tearDown = helpers.HTTPServerTest.tearDown
    request = helpers.HTTPServerTest.request
    json = staticmethod(helpers.HTTPServerTest.json)

    def configure_catalog(self):
        self.backend.variation_contract = lambda: contract(None)
        self.backend.select_variation = lambda payload: select(payload, spacing_m=.000625)
        self.resets = []
        original = self.backend.reset

        def reset(*, seed=None, options=None):
            self.resets.append({"seed": seed, "options": options})
            return original(seed=seed, options=options)

        self.backend.reset = reset

    def test_preview_does_not_reset_and_start_retry_is_idempotent(self):
        self.configure_catalog()
        payload = {"constraints": {"aliquot_ml": 12., "burette_ml": 22.}}
        preview = self.json(self.request("/v1/variations/select", method="POST", body=payload))
        self.assertFalse(preview["committed"])
        self.assertEqual(preview["reagents"]["aliquot_ml"], 12.)
        self.assertEqual(preview["reagents"]["burette_ml"], 22.)
        self.assertEqual(self.resets, [])
        first = self.json(self.request("/v1/variations/start", method="POST", body=payload,
                                       idempotency_key="catalog-start"))
        second = self.json(self.request("/v1/variations/start", method="POST", body=payload,
                                        idempotency_key="catalog-start"))
        self.assertEqual(first, second)
        self.assertEqual(len(self.resets), 1)
        self.assertEqual(self.resets[0]["options"], preview["reset_request"]["options"])

    def test_unsupported_exact_constraint_does_not_reset(self):
        self.configure_catalog()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/variations/start", method="POST",
                         body={"constraints": {"aliquot_ml": 12.1}})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.resets, [])
