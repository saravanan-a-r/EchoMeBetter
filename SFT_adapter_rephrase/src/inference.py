"""
Using a trained adapter: base + adapter in, rewrites out.

    from src.inference import RephraseAdapter

    rephraser = RephraseAdapter.load("SFT_adapter_rephrase/runs/<run>/final")
    rephraser.rephrase(["Please make sure that you submit the report by Friday."])

The base checkpoint defaults to the one recorded in the adapter, and the
adapter refuses to attach to any other weights (fingerprint check).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .adapter_io import load_adapter, read_adapter_config
from .base_model import load_base_model
from .batching import collate
from .codec import TextCodec
from .config import DecodingConfig
from .data import SFTExample
from .decoding import decode_batch
from .errors import AdapterError
from .interop import DEFAULT_TOKENIZER_DIR

DEFAULT_DECODING = DecodingConfig(
    strategy="greedy", num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, max_new_tokens=384
)


class RephraseAdapter:
    def __init__(self, model, codec: TextCodec, device: torch.device) -> None:
        self.model = model.eval()
        self.codec = codec
        self.device = device

    @classmethod
    def load(
        cls,
        adapter_dir: str | Path,
        *,
        base_checkpoint: str | Path | None = None,
        tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
        device: str = "cpu",
    ) -> "RephraseAdapter":
        document = read_adapter_config(adapter_dir)
        style_token = document.get("style_token")
        if not style_token:
            raise AdapterError(f"{adapter_dir} does not record its style token")
        codec = TextCodec.from_tokenizer_dir(tokenizer_dir, style_token)
        model, _ = load_base_model(Path(base_checkpoint or document["base"]["checkpoint"]))
        load_adapter(model, adapter_dir, token_map=codec.token_map)
        torch_device = torch.device(device)
        return cls(model.to(torch_device), codec, torch_device)

    @torch.no_grad()
    def rephrase(
        self, texts: Sequence[str], *, decoding: DecodingConfig = DEFAULT_DECODING, batch_size: int = 16
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            examples = [
                SFTExample(
                    key="",
                    source_text=text,
                    target_text="",
                    encoder_input_ids=tuple(self.codec.frame.source(self.codec.encode(text))),
                    target_ids=(self.codec.frame.eos_id,),
                )
                for text in chunk
            ]
            batch = collate(examples, self.codec.frame).to(self.device)
            result = decode_batch(self.model, batch.encoder_input_ids, batch.encoder_attention_mask, decoding)
            outputs.extend(self.codec.decode(ids).strip() for ids in result.sequences)
        return outputs
