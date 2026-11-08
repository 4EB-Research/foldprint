from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from protein_sae_pipeline.representations.biohub import (
    BiohubClient,
    BiohubUnavailableError,
    live_biohub_configured,
)


class OptionalBiohubTests(unittest.TestCase):
    def test_offline_import_does_not_import_esm_sdk(self) -> None:
        self.assertNotIn("esm.sdk.api", sys.modules)

    def test_missing_token_fails_before_optional_sdk_import(self) -> None:
        with patch.dict(os.environ, {"BIOHUB_API_TOKEN": ""}, clear=False):
            self.assertFalse(live_biohub_configured())
            with self.assertRaisesRegex(BiohubUnavailableError, "requires a token"):
                BiohubClient()
        self.assertNotIn("esm.sdk.api", sys.modules)


if __name__ == "__main__":
    unittest.main()
