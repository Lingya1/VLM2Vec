"""E4c: does the register substate accumulate counterpart information *inside* the loop?

§5.7 and §5.8 answer a different question than they appear to. They vary the number of loops
and read the state after the suffix layer, so what they measure is the deployed embedding at
several inference depths -- and for the depth-1 checkpoint, every depth but the first is
outside the regime the model was trained in. Two consequences a reviewer identified and we
could not answer from that data:

  1. the significant decline is carried by rounds past the trained depth, so it is partly the
     depth-mismatch phenomenon of §5.5 rather than a statement about what the loop does;
  2. the readout there is post-suffix, so it cannot separate "the loop failed to accumulate"
     from "the suffix discarded what the loop accumulated."

This script measures the states inside a *single* forward pass at the trained depth. The
schedule replays layers, so output_hidden_states yields one state per application and the loop
boundaries fall at known indices; the state after loop iteration k is available without
changing the computation at all. Nothing here is off-depth, and nothing passes the suffix.

The accumulation account makes a directional prediction about exactly this quantity: the
register substate should carry more about the counterpart after iteration k+1 than after k.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/zhoutuowen/VLM2Vec")
sys.path.insert(0, "/home/zhoutuowen/VLM2Vec/experiments/public/train")

from probe_layerwise_separation import _open_images, load_pairs
from probe_usable_information import (find_schedule, select_alpha_across_rounds,
                                      usable_information)
from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, load_processor, process_vlm_inputs_fns
from src.utils.basic_utils import batch_to_device


@torch.no_grad()
def encode_iterations(model, processor, items, data_args, batch_size, desc, boundaries, n_reg):
    """Features at each loop boundary of one forward pass, plus the post-suffix state."""
    process_fn = process_vlm_inputs_fns[QWEN2_VL]
    nrm = lambda v: torch.nn.functional.normalize(v, p=2, dim=-1)
    out = {k: [[] for _ in range(len(boundaries) + 1)]
           for k in ("ro", "reg", "regmean", "cont")}

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = {"text": [t for t, _ in chunk],
                 "images": [_open_images(p) for _, p in chunk]}
        inputs = process_fn(batch, processor=processor, max_length=data_args.max_len)
        inputs = batch_to_device(inputs, model.device)

        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            model_input = model._prepare_model_input(inputs)
            if model.reloop is not None:
                model_input['input_ids'], model_input['attention_mask'] = \
                    model.reloop.extend_inputs(model_input['input_ids'],
                                               model_input['attention_mask'])
            res = model.encoder(**model_input, return_dict=True, output_hidden_states=True)
            states = res.hidden_states
            mask = model_input['attention_mask'].float()

        for slot, hidx in enumerate(list(boundaries) + [len(states) - 1]):
            h = states[hidx].float()
            content = h[:, :-n_reg, :] if n_reg else h
            cmask = mask[:, :-n_reg] if n_reg else mask
            denom = cmask.sum(1, keepdim=True).clamp(min=1)
            out["ro"][slot].append(nrm(h[:, -1, :]).cpu())
            out["cont"][slot].append(nrm((content * cmask.unsqueeze(-1)).sum(1) / denom).cpu())
            if n_reg:
                reg = h[:, -n_reg:, :]
                out["reg"][slot].append(nrm(reg.reshape(h.shape[0], -1)).cpu())
                out["regmean"][slot].append(nrm(reg.mean(dim=1)).cpu())

        if (start // batch_size) % 10 == 0:
            print(f"  {desc}: {min(start + batch_size, len(items))}/{len(items)}",
                  end="", flush=True)
    print()
    return {k: [torch.cat(s).numpy().astype(np.float64) for s in v if s]
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--model_name",
                    default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--subset", default="OK-VQA")
    ap.add_argument("--num_pairs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--n_pca", type=int, default=64)
    ap.add_argument("--image_dir", default="/home/zhoutuowen/data/MMEB-train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_args = DataArguments()
    data_args.max_len = 1024
    # Same subset, count and seed as E4, so the pairs and the split are the identical ones.
    queries, candidates = load_pairs(args.subset, args.num_pairs, args.image_dir, args.seed)

    state = torch.load(os.path.join(args.checkpoint_path, "reloop.pt"),
                       map_location="cpu", weights_only=False)
    import glob as _glob
    is_fullft = (os.path.exists(os.path.join(args.checkpoint_path, "model.safetensors"))
                 or bool(_glob.glob(os.path.join(args.checkpoint_path, "model-*.safetensors"))))
    model_args = ModelArguments(
        model_name=args.checkpoint_path if is_fullft else args.model_name,
        checkpoint_path=args.checkpoint_path, model_backbone=QWEN2_VL,
        lora=not is_fullft, pooling="eos", normalize=True,
        reloop_t=state["reloop_t"], reloop_m=state["reloop_m"],
        reloop_loop_start=state["reloop_loop_start"],
        reloop_loop_end=state["reloop_loop_end"])
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    sched, _ = find_schedule(model)
    T, width = sched.num_loops, sched.loop_end - sched.loop_start
    boundaries = [sched.loop_start + width * k for k in range(1, T + 1)]
    n_reg = model.reloop.num_registers if model.reloop is not None else 0
    print(f"trained depth T={T}; loop boundaries at hidden_states{boundaries}; "
          f"{n_reg} registers. Depth is NOT varied.")

    # The target is the counterpart under the untuned base model, exactly as in E4, so the two
    # measurements are on the same scale.
    base_args = ModelArguments(model_name=args.model_name, model_backbone=QWEN2_VL,
                               lora=False, pooling="eos", normalize=True)
    base = MMEBModel.load(base_args, is_trainable=False,
                          processor=load_processor(base_args, data_args))
    base = base.to("cuda", dtype=torch.bfloat16).eval()
    Y = encode_iterations(base, processor, candidates, data_args, args.batch_size,
                          "Y", [], 0)["ro"][0]
    del base
    torch.cuda.empty_cache()

    n_train = int(args.train_frac * len(queries))
    n_sub = n_train - max(1, int(0.25 * n_train))
    ymean = Y[:n_sub].mean(0)
    _, S, Vt = np.linalg.svd(Y[:n_sub] - ymean, full_matrices=False)
    n_pca = min(args.n_pca, Vt.shape[0])
    Yp = (Y - ymean) @ Vt[:n_pca].T
    # Fingerprint the target so that "the same Y as E4" is a checkable claim rather than an
    # inference from two scripts naming the same model. They resolved the base weights through
    # different paths, and we had no way to confirm the targets matched beyond a statistic
    # printed to three decimals.
    y_fp = hashlib.sha256(np.round(Y, 6).tobytes()).hexdigest()[:16]
    print(f"Y -> PCA {Yp.shape}, sub-train variance retained "
          f"{float((S[:n_pca]**2).sum()/(S**2).sum()):.6f}, Y fingerprint {y_fp}")

    feats = encode_iterations(model, processor, queries, data_args, args.batch_size,
                              f"T={T} query", boundaries, n_reg)

    alphas = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    labels = [f"after loop {k}" for k in range(1, T + 1)] + ["after suffix (deployed)"]
    rows = [{"stage": s} for s in labels]
    resid, base_resid, chosen = {}, {}, {}
    for tag in ("ro", "reg", "regmean", "cont"):
        if not len(feats[tag][0]):
            continue
        chosen[tag] = select_alpha_across_rounds(feats[tag], Yp, n_train, alphas)
        for row, X in zip(rows, feats[tag]):
            nats, r2, alpha, se, per_item, bpi = usable_information(X, Yp, n_train,
                                                                    chosen[tag])
            row.update({f"{tag}_nats": nats, f"{tag}_r2": r2, f"{tag}_se": se})
            resid.setdefault(tag, []).append(per_item)
            base_resid[tag] = bpi

    print("\nshared penalty per family: "
          + ", ".join(f"{t}={a:g}" for t, a in chosen.items()))
    print(f"\n{'stage':<26}{'readout':>9}{'reg x5':>10}{'reg mean':>10}{'content':>9}")
    for row in rows:
        print(f"{row['stage']:<26}{row['ro_nats']:>9.3f}{row['reg_nats']:>10.3f}"
              f"{row['regmean_nats']:>10.3f}{row['cont_nats']:>9.3f}")

    # The prediction under test is directional and specified before looking: the register
    # substate should be larger after the last loop iteration than after the first.
    rng = np.random.default_rng(0)
    n_te, n_boot = len(base_resid["ro"]), 10000
    idx = rng.integers(0, n_te, size=(n_boot, n_te))
    paired = {}
    for tag, rs in resid.items():
        b = base_resid[tag]
        f = lambda pi: np.array([0.5 * np.log(b[i].mean(0)
                                              / np.maximum(pi[i].mean(0), 1e-12)).sum()
                                 for i in idx])
        d = f(rs[T - 1]) - f(rs[0])
        n_le = int((d <= 0).sum())
        paired[tag] = {"contrast": "after loop 1 -> after final loop",
                       "observed": rows[T - 1][f"{tag}_nats"] - rows[0][f"{tag}_nats"],
                       "se": float(np.std(d)),
                       "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       "n_boot": n_boot, "p_le_0": (n_le + 1) / (n_boot + 1)}

    print("\nwithin one trained forward pass, loop 1 -> final loop (accumulation predicts > 0)")
    for tag, p in paired.items():
        print(f"  {tag:>8s}  observed={p['observed']:>+7.3f}  se={p['se']:.3f}  "
              f"95% CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  "
              f"P(delta<=0)={p['p_le_0']:.4f}")

    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint_path, "subset": args.subset,
                   "trained_t": T, "boundaries": boundaries, "num_pairs": len(queries),
                   "n_train": n_train, "n_pca": n_pca, "y_fingerprint": y_fp, "depth_varied": False,
                   "paired_first_to_last_loop": paired, "rows": rows}, f, indent=2)
    np.savez_compressed(args.output.replace(".json", "_resid.npz"),
                        **{f"{t}_s{i}": r for t, rs in resid.items() for i, r in enumerate(rs)},
                        **{f"{t}_base": v for t, v in base_resid.items()})
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
