"""Top-k sparsification + fp16 packing for gradient deltas.

Wire format (little-endian), one block per named tensor:
  [u16 name_len][name utf-8]
  [u32 numel_original][u32 k]
  [k * u32 indices][k * f16 values]

~100x reduction at k = 1% of numel. Error feedback (residual accumulation)
is kept on the worker side so dropped coordinates aren't lost, just delayed.
"""
import base64
import struct
from typing import Dict

import numpy as np
import torch

TOPK_FRACTION = 0.01  # keep top 1% of coordinates by magnitude


def compress_state_delta(delta: Dict[str, torch.Tensor]) -> str:
    """delta: name -> dense fp32 tensor (local_params - global_params)."""
    out = bytearray()
    for name, t in delta.items():
        flat = t.detach().float().flatten()
        numel = flat.numel()
        k = max(1, int(numel * TOPK_FRACTION))
        vals, idx = torch.topk(flat.abs(), k)
        idx = idx.to(torch.int64)
        picked = flat[idx]

        name_b = name.encode()
        out += struct.pack("<H", len(name_b)) + name_b
        out += struct.pack("<II", numel, k)
        out += idx.to(torch.int32).numpy().astype("<u4").tobytes()
        out += picked.to(torch.float16).numpy().astype("<f2").tobytes()
    return base64.b64encode(bytes(out)).decode()


def decompress_state_delta(payload_b64: str) -> Dict[str, torch.Tensor]:
    """Returns name -> dense fp32 flat tensor. Caller reshapes against the model."""
    buf = base64.b64decode(payload_b64)
    pos, result = 0, {}
    while pos < len(buf):
        (name_len,) = struct.unpack_from("<H", buf, pos); pos += 2
        name = buf[pos:pos + name_len].decode(); pos += name_len
        numel, k = struct.unpack_from("<II", buf, pos); pos += 8
        idx = np.frombuffer(buf, dtype="<u4", count=k, offset=pos); pos += 4 * k
        vals = np.frombuffer(buf, dtype="<f2", count=k, offset=pos); pos += 2 * k

        dense = torch.zeros(numel, dtype=torch.float32)
        dense[torch.from_numpy(idx.astype(np.int64))] = torch.from_numpy(
            vals.astype(np.float32)
        )
        result[name] = dense
    return result
