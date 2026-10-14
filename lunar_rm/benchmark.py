"""Inference-only timing with explicit cache, transfer, and memory boundaries."""

import platform
import resource
import time

import numpy as np
import torch
from torch import nn

from .models import parameter_report


def _measure(call, sync, warmup, repeats):
    for _ in range(warmup):
        call()
    sync()
    values = []
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        call()
        sync()
        values.append((time.perf_counter()-started)*1000)
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "repeats": repeats, "samples_ms": values}


@torch.inference_mode()
def benchmark_model(model, dataset, index, *, device="cpu", warmup=3, repeats=10, amp=False):
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("positive warmup/repeats required")
    if type(index) is not int or not 0 <= index < len(dataset):
        raise ValueError("benchmark index outside dataset")
    if dataset.augment:
        raise ValueError("benchmark requires unaugmented data")
    if amp and not str(device).startswith("cuda"):
        raise ValueError("AMP benchmark requires CUDA")
    dev = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    bi, ri = divmod(index, len(dataset.rows))
    band, row = dataset.bands[bi], dataset.rows[ri]
    # Setup reads only input DEM, never labels or masks. I/O is outside timing.
    dem = np.array(dataset._array("hm")[int(row["hm_row"])], dtype=np.float32, copy=True)
    tx = (int(row["tx_row"]), int(row["tx_col"]))
    old_cache, old_need_los, old_training = dataset.cache, dataset.need_los, model.training
    model.to(dev).eval()
    dataset.cache, dataset.need_los = None, False

    def inputs():
        return dataset.build_inputs(dem, tx, row, band)

    def transfer(item):
        x = item["features"][None].to(dev)
        base = item.get("baseline_db")
        return x, base[None].to(dev) if base is not None else None

    def forward(pair):
        with torch.autocast(device_type=dev.type, enabled=amp):
            return model(*pair)

    def end_to_end():
        # Includes CPU feature assembly, H2D and output D2H; excludes disk I/O.
        return forward(transfer(inputs())).cpu()

    handles, conv_macs = [], [0]
    try:
        pair = transfer(inputs())
        if not torch.isfinite(forward(pair)).all():
            raise ValueError("non-finite benchmark prediction")
        if dev.type == "cuda":
            sync()
            torch.cuda.reset_peak_memory_stats(dev)
        timings = {"feature_generation": _measure(inputs, lambda: None, warmup, repeats),
                   "network_forward": _measure(lambda: forward(pair), sync, warmup, repeats),
                   "end_to_end": _measure(end_to_end, sync, warmup, repeats)}
        gpu_memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(dev),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev),
                       "scope": "all uncached timing lanes including warmup; model and tensors included"}
                      if dev.type == "cuda" else None)
        timings["cache_hit_inputs"] = None
        if old_cache is not None:
            dataset.cache = old_cache
            inputs()  # populate outside cache-hit timing
            timings["cache_hit_inputs"] = _measure(inputs, lambda: None, warmup, repeats)
            dataset.cache = None

        def count_conv(module, args, output):
            conv_macs[0] += output.numel() * (module.in_channels // module.groups) * int(np.prod(module.kernel_size))

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(count_conv))
        forward(pair)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"batch_size": 1, "sample_id": row["sample_id"], "band": band,
                "warmup": warmup, "uncached": True, "training_started": False,
                "weights": "caller-supplied; see checkpoint identity", "timings": timings,
                "parameters": parameter_report(model),
                "complexity": {"conv_macs": conv_macs[0], "conv_flops": 2*conv_macs[0],
                               "complete": False, "convention": "Conv2d multiply-add = 1 MAC = 2 FLOPs",
                               "excluded": ["bias", "normalization", "activation", "pooling", "interpolation",
                                            "concatenation", "output affine/residual", "physical features", "transfers"]},
                "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                             "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                             "torch": torch.__version__, "threads": torch.get_num_threads(),
                             "interop_threads": torch.get_num_interop_threads(),
                             "precision": "autocast default CUDA fp16; output fp32" if amp else "fp32",
                             "feature_dtype": dataset.feature_dtype, "cuda_synchronized": dev.type == "cuda"},
                "boundaries": {"feature_generation": "CPU raw DEM -> model tensors; no disk or feature cache",
                               "network_forward": "device-resident inputs to device-resident output",
                               "end_to_end": "CPU in-memory DEM -> features -> H2D -> forward -> CPU dB map; disk I/O excluded",
                               "cache_hit_inputs": "CPU input preparation with warmed physical feature cache; excludes network",
                               "scope": "single fixed sample; not dataset-wide latency distribution"},
                "cpu_memory": {"process_lifetime_peak_rss_bytes": int(rss * (1 if platform.system() == "Darwin" else 1024)),
                               "scope": "process lifetime high-water RSS, not isolated benchmark allocation"},
                "gpu_memory": gpu_memory,
                "end_to_end_target_500ms_met": timings["end_to_end"]["median_ms"] < 500}
    finally:
        for handle in handles:
            handle.remove()
        dataset.cache, dataset.need_los = old_cache, old_need_los
        model.train(old_training)
