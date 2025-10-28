from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel

# Import custom Geneformer tokenizer
from geneformer import (
    TranscriptomeTokenizer,
    TOKEN_DICTIONARY_FILE,
    ENSEMBL_DICTIONARY_FILE,
)
import pickle


logger = logging.getLogger(__name__)


def _seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GFEmbedder:
    """Helper for Geneformer V2 embeddings (CLS token).

    Loads tokenizer and model from a model repo with subfolder support.
    Provides conversion from AnnData to rank tokens and batched CLS embedding.
    If GF_OFFLINE environment variable is set to "1", returns deterministic
    pseudo-embeddings without downloading the model, for smoke testing.
    """

    def __init__(
        self,
        model_repo: str | None = None,
        submodel: str | None = None,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.model_repo = model_repo or os.getenv("GF_MODEL_REPO", "ctheodoris/Geneformer")
        self.submodel = submodel or os.getenv("GF_SUBMODEL", "Geneformer-V2-104M")
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        _seed_everything(seed)

        self.offline = os.getenv("GF_OFFLINE", "0") == "1"
        self.tokenizer = None
        self.model = None

        if not self.offline:
            logger.info("Loading tokenizer and model: %s / %s", self.model_repo, self.submodel)
            # Use the pre-installed tokenizer dictionary
            self.tokenizer = TranscriptomeTokenizer()
            # Load gene name -> Ensembl ID dictionary to map symbols to tokenizer keys
            try:
                with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
                    name_to_id = pickle.load(f)
                # Normalize keys to uppercase strings for robust matching
                self.gene_name_to_ensembl = {str(k).upper(): v for k, v in name_to_id.items()}
            except Exception:
                self.gene_name_to_ensembl = {}
            # Load the model as before
            self.model = AutoModel.from_pretrained(self.model_repo, subfolder=self.submodel)
            # Prefer FP16 on CUDA for speed
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            if self.device.type == "cuda":
                self.model.to(self.device, dtype=torch.float16)
            else:
                self.model.to(self.device)
            self.model.eval()
        else:
            logger.warning("GF_OFFLINE=1 -> using pseudo-embeddings for smoke tests.")
            self.gene_name_to_ensembl = {}

    def _get_vocab_set(self) -> set[str]:
        if self.offline:
            # Minimal fake vocab set for smoke tests
            return {"GENE_FAKE_A", "GENE_FAKE_B", "GENE_FAKE_C"}
        assert self.tokenizer is not None
        # Geneformer tokenizer exposes gene vocabulary via gene_token_dict / gene_keys
        if hasattr(self.tokenizer, "gene_keys"):
            return set(self.tokenizer.gene_keys)
        return set(getattr(self.tokenizer, "gene_token_dict", {}).keys())

    def _pad_token_id(self) -> int:
        if self.offline:
            return 0
        assert self.tokenizer is not None
        vocab_dict = getattr(self.tokenizer, "gene_token_dict", {})
        if not vocab_dict:
            return 0
        # Prefer explicit pad, otherwise fall back to eos, else arbitrary first id
        return (
            vocab_dict.get("<pad>")
            or vocab_dict.get("<eos>")
            or next(iter(vocab_dict.values()))
        )

    def adata_to_rank_tokens(self, adata, top_k: int = 4096) -> List[List[int]]:
        """Convert AnnData expression to rank tokens for Geneformer.

        Sort genes per cell by descending expression and keep top_k present in tokenizer vocab.

        Args:
          adata: AnnData with X or layers["counts"].
          top_k: Maximum number of tokens per cell.

        Returns:
          List of token id lists, one per cell.
        """
        X = adata.layers.get("counts", adata.X)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        n_cells, n_genes = X.shape
        # Prefer explicit gene symbol column; otherwise use var_names
        gene_symbols = adata.var.get("gene_symbol", adata.var_names).astype(str).tolist()

        # Map gene symbols to tokenizer keys (Ensembl IDs) when needed
        if not self.offline:
            mapped_gene_ids: List[str] = []
            for sym in gene_symbols:
                # If already an Ensembl-like ID, keep it; else map via dictionary
                if sym.upper().startswith("ENSG"):
                    mapped_gene_ids.append(sym)
                else:
                    mapped_gene_ids.append(self.gene_name_to_ensembl.get(sym.upper(), sym))
        else:
            mapped_gene_ids = gene_symbols

        vocab = self._get_vocab_set()
        if not self.offline:
            assert self.tokenizer is not None

        # If a rank-based perturbation was applied, use the provided rank matrix
        rank_matrix = adata.obsm.get("rank_matrix", None)

        tokens: List[List[int]] = []
        for i in range(n_cells):
            if rank_matrix is not None:
                # Lower rank index = higher priority; argsort ascending
                order = np.argsort(rank_matrix[i])
            else:
                # Default: descending expression order
                order = np.argsort(-X[i])
            kept: List[int] = []
            for j in order:
                g_id = mapped_gene_ids[j]
                if g_id in vocab:
                    if self.offline:
                        kept.append(abs(hash(g_id)) % 10000)
                    else:
                        # Map Ensembl ID to token id via gene_token_dict
                        token_id = self.tokenizer.gene_token_dict.get(g_id)
                        if token_id is not None:
                            kept.append(token_id)
                if len(kept) >= top_k:
                    break
            if len(kept) == 0:
                # Fallback to a dummy token id to avoid empty sequences
                kept = [0]
            tokens.append(kept)
        return tokens

    @torch.no_grad()
    def get_cls_embeddings(self, token_batches: Sequence[Sequence[int]], batch_size: int = 16) -> np.ndarray:
        """Run model forward pass to get CLS embeddings.

        Args:
          token_batches: list of token id lists (variable length).
          batch_size: micro-batch size.

        Returns:
          ndarray [n_cells, hidden_dim].
        """
        if self.offline:
            # Produce deterministic pseudo-embeddings
            max_len = max(len(t) for t in token_batches)
            n = len(token_batches)
            hidden = 768
            out = np.zeros((n, hidden), dtype=np.float32)
            for i, toks in enumerate(token_batches):
                rnd = np.random.RandomState(len(toks) % 123 + 7)
                out[i] = rnd.normal(0, 1, size=hidden).astype(np.float32)
            return out

        assert self.tokenizer is not None and self.model is not None
        model = self.model
        device = self.device

        # Pad sequences to max length per batch
        def pad_to_max(seqs: List[List[int]]) -> torch.Tensor:
            maxlen = max(len(s) for s in seqs)
            pad_id = self._pad_token_id()
            padded = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
            return torch.tensor(padded, dtype=torch.long, device=device)

        all_out: List[np.ndarray] = []
        for i in range(0, len(token_batches), batch_size):
            chunk = token_batches[i : i + batch_size]
            input_ids = pad_to_max([list(s) for s in chunk])
            attention_mask = (input_ids != self._pad_token_id()).long()
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Assuming outputs.last_hidden_state with CLS at index 0
            hidden_states = outputs.last_hidden_state  # [B, T, H]
            cls_vec = hidden_states[:, 0, :].float().detach().cpu().numpy()
            all_out.append(cls_vec)
        return np.concatenate(all_out, axis=0)


