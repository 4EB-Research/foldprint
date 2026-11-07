"""Optional live Biohub inference adapter.

Nothing in this module imports the ESM SDK, reads a token, or accesses the
network at import time.  The public example consumes stored fictional
representations and therefore never constructs :class:`BiohubClient`.  Live
scoring is an explicitly optional path for user-supplied sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
import sys
from typing import Any

import numpy as np

from .pooling import mean_pool_layer60, residue_hidden_states
from .sae import (
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_TOPK,
    SAENormalization,
    extract_topk_residue_activations,
    normalize_sae_activations,
)
from .sparse import SparseResidueMatrix, SparseVector


DEFAULT_ESMC_MODEL = "esmc-6b-2024-12"
DEFAULT_SAE_MODEL = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"


class BiohubUnavailableError(RuntimeError):
    """Raised only when an explicitly requested live Biohub path is unavailable."""


@dataclass(frozen=True)
class BiohubConfig:
    url: str = "https://biohub.ai"
    token_env: str = "BIOHUB_API_TOKEN"
    esmc_model: str = DEFAULT_ESMC_MODEL
    sae_model: str = DEFAULT_SAE_MODEL
    hidden_layer: int = 60
    embedding_dimension: int = 2_560
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    topk: int = DEFAULT_TOPK

    def __post_init__(self) -> None:
        if not self.url or not self.token_env or not self.esmc_model or not self.sae_model:
            raise ValueError("Biohub URL, token variable, and model names must be non-empty")
        if self.hidden_layer != 60:
            raise ValueError("the pipeline is pinned to ESM-C hidden layer 60")
        if self.embedding_dimension <= 0 or self.codebook_size <= 0:
            raise ValueError("representation dimensions must be positive")
        if self.topk <= 0 or self.topk > self.codebook_size:
            raise ValueError("SAE TopK must be in the codebook range")


@dataclass(frozen=True)
class Layer60Result:
    residue_states: np.ndarray
    mean_embedding: np.ndarray


@dataclass(frozen=True)
class SAEResult:
    residue_activations: SparseResidueMatrix
    max_pooled_profile: SparseVector


def biohub_sdk_available() -> bool:
    """Return whether the optional SDK appears installed, without importing it."""

    return importlib.util.find_spec("esm") is not None


def live_biohub_configured(*, token_env: str = "BIOHUB_API_TOKEN") -> bool:
    """Check local opt-in prerequisites; this does not validate a token online."""

    return bool(os.environ.get(token_env, "").strip()) and biohub_sdk_available()


def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _trim_sae_tokens(value: Any, *, sequence_length: int, codebook_size: int) -> np.ndarray:
    matrix = _dense_float32(value)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[1] != codebook_size:
        raise ValueError(
            f"Biohub SAE output must be token-by-{codebook_size}; received {matrix.shape}"
        )
    if matrix.shape[0] != sequence_length + 2:
        raise ValueError(
            "Biohub SAE output must contain BOS + residues + EOS: "
            f"tokens={matrix.shape[0]}, expected={sequence_length + 2}"
        )
    residues = matrix[1 : sequence_length + 1]
    if not np.isfinite(residues).all() or np.any(residues < 0):
        raise ValueError("Biohub SAE residue activations are invalid")
    return residues


class BiohubClient:
    """Lazy adapter for live ESM-C layer-60 and normalized SAE requests.

    Constructing this object is the opt-in boundary: only then is the token
    read and the optional SDK imported.  Tokens are never stored in the config,
    represented in output metadata, or included in exception messages.
    """

    def __init__(self, config: BiohubConfig | None = None, *, token: str | None = None):
        self.config = config or BiohubConfig()
        resolved_token = (
            token if token is not None else os.environ.get(self.config.token_env, "")
        ).strip()
        if not resolved_token:
            raise BiohubUnavailableError(
                f"live Biohub scoring requires a token in {self.config.token_env}"
            )
        try:
            # Import scipy first to avoid duplicate OpenMP runtime initialization on macOS.
            importlib.import_module("scipy.linalg")
            api = importlib.import_module("esm.sdk.api")
            forge = importlib.import_module("esm.sdk.forge")
        except ImportError as error:
            raise BiohubUnavailableError(
                "the optional Biohub ESM SDK is not installed for Python "
                f"{sys.version_info.major}.{sys.version_info.minor}; install the 'biohub' extra"
            ) from error

        self._ESMProtein = api.ESMProtein
        self._ESMProteinError = api.ESMProteinError
        self._LogitsConfig = api.LogitsConfig
        self._SAEConfig = api.SAEConfig
        self._client = forge.ESMCForgeInferenceClient(
            model=self.config.esmc_model,
            url=self.config.url,
            token=resolved_token,
        )

    def _raise_if_error(self, value: Any) -> None:
        if isinstance(value, self._ESMProteinError):
            raise RuntimeError(f"Biohub error {value.error_code}: {value.error_msg}")

    def encode(self, sequence: str) -> Any:
        if not sequence:
            raise ValueError("cannot encode an empty protein sequence")
        encoded = self._client.encode(self._ESMProtein(sequence=sequence))
        self._raise_if_error(encoded)
        if len(encoded) != len(sequence) + 2:
            raise ValueError(
                "Biohub encoding is not BOS + residues + EOS: "
                f"tokens={len(encoded)}, expected={len(sequence) + 2}"
            )
        return encoded

    def infer_layer60(self, sequence: str, *, encoded: Any | None = None) -> Layer60Result:
        """Request hidden layer 60, remove BOS/EOS, and mean pool in float32."""

        protein = self.encode(sequence) if encoded is None else encoded
        if len(protein) != len(sequence) + 2:
            raise ValueError("provided Biohub encoding length does not match sequence")
        output = self._client.logits(
            protein,
            config=self._LogitsConfig(
                return_hidden_states=True,
                ith_hidden_layer=self.config.hidden_layer,
            ),
            return_bytes=True,
        )
        self._raise_if_error(output)
        if output.hidden_states is None:
            raise ValueError("Biohub response did not contain layer-60 hidden states")
        residues = residue_hidden_states(
            output.hidden_states,
            sequence_length=len(sequence),
            expected_dimension=self.config.embedding_dimension,
        )
        mean = mean_pool_layer60(
            output.hidden_states,
            sequence_length=len(sequence),
            expected_dimension=self.config.embedding_dimension,
        )
        return Layer60Result(residues, mean)

    def infer_sae(
        self,
        sequence: str,
        *,
        normalization: SAENormalization | None = None,
        encoded: Any | None = None,
        request_server_normalized: bool = False,
    ) -> SAEResult:
        """Request SAE activations, normalize locally, and construct the max pool.

        The supported path leaves ``request_server_normalized=False`` and supplies
        the pinned normalization arrays. This requests raw magnitudes and
        applies the audited divide-then-IDF operations locally. Server-side
        normalization is available only for an explicit diagnostic comparison.
        """

        if request_server_normalized and normalization is not None:
            raise ValueError(
                "do not combine server-side normalization with local normalization assets"
            )
        if not request_server_normalized and normalization is None:
            raise ValueError("SAE inference requires pinned local normalization assets")
        if normalization is not None and normalization.codebook_size != self.config.codebook_size:
            raise ValueError("SAE normalization assets do not match the configured codebook")
        protein = self.encode(sequence) if encoded is None else encoded
        if len(protein) != len(sequence) + 2:
            raise ValueError("provided Biohub encoding length does not match sequence")
        output = self._client.logits(
            protein,
            config=self._LogitsConfig(
                sae_config=self._SAEConfig(
                    models=[self.config.sae_model],
                    normalize_features=request_server_normalized,
                )
            ),
            return_bytes=False,
        )
        self._raise_if_error(output)
        if self.config.sae_model not in output.sae_outputs:
            raise ValueError(f"Biohub response omitted SAE model {self.config.sae_model}")
        dense = _trim_sae_tokens(
            output.sae_outputs[self.config.sae_model],
            sequence_length=len(sequence),
            codebook_size=self.config.codebook_size,
        )
        if normalization is not None:
            dense = normalize_sae_activations(dense, normalization)
        residues = extract_topk_residue_activations(dense, topk=self.config.topk)
        return SAEResult(residues, residues.max_pool())
