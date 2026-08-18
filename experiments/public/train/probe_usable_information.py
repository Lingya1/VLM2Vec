"""E4: estimate usable information u_t and total information c_t per recurrent round.

The paper's frame says depth cannot raise the ceiling c_t = I(H^(t); Y) but can raise the
fraction u_t = I_V(H^(t) -> Y) that a deployed readout reaches. Four drafts asserted this
without measuring it. This script measures it.

Y is the counterpart's *content* encoding under a frozen encoder (the untuned base model),
never its identity among candidates -- so no candidate pool enters and the quantity does not
inherit the batch-relativity the paper objects to elsewhere.

Two predictor families over the same state H^(t):

  V_ro    affine from the single readout position   -> matches what cosine scoring can exploit
  V_rich  affine from mean/max/std over all content positions plus the readout

Both are fit by ridge on a train split and scored on held-out pairs, so the reported numbers
are honest generalization estimates rather than fit quality. Under a diagonal Gaussian
predictor the usable information in nats is 0.5 * sum_j log(Var_j(Y) / MSE_j), which is what
we report; it lower-bounds the corresponding V-information.

V_rich is a *proxy* for the ceiling, not the ceiling. It is a larger family, so it estimates a
larger usable information, but an unconstrained family is not available at finite sample size.
The prediction under test is therefore directional: u_t should rise while c_t does not.
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
from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, load_processor, process_vlm_inputs_fns
from src.utils.basic_utils import batch_to_device


def find_schedule(model):
    for name, mod in model.named_modules():
        if hasattr(mod, "recurrence") and mod.recurrence is not None:
            return mod.recurrence, name
    raise RuntimeError("no recurrence schedule found")


@torch.no_grad()
def encode_states(model, processor, items, data_args, batch_size, desc, want_rich):
    """Return (readout, rich) features. rich is None when want_rich is False.

    The readout is taken exactly as the deployed system takes it, so that u_t is estimated on
    the same vector retrieval actually scores. The rich features pool over content positions
    only -- register positions are excluded, since including them would let V_rich see the
    readout twice and inflate the ceiling estimate.
    """
    process_fn = process_vlm_inputs_fns[QWEN2_VL]
    ro_out, rich_out, reg_out, regmean_out, cont_out = [], [], [], [], []
    n_reg = model.reloop.num_registers if model.reloop is not None else 0

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
            out = model.encoder(**model_input, return_dict=True, output_hidden_states=True)
            h = out.hidden_states[-1].float()
            mask = model_input['attention_mask'].float()

        # The readout is L2-normalized because the deployed system normalizes it. Every other
        # block is normalized the same way, so that V_rich literally contains V_ro's columns:
        # an earlier version concatenated the *un*normalized readout, which meant the larger
        # family did not nest the smaller one and duly scored below it.
        nrm = lambda v: torch.nn.functional.normalize(v, p=2, dim=-1)
        ro = nrm(h[:, -1, :])
        ro_out.append(ro.cpu())

        content = h[:, :h.shape[1] - n_reg, :] if n_reg else h
        m = mask[:, :mask.shape[1] - n_reg] if n_reg else mask
        m = m.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (content * m).sum(dim=1) / denom
        var = ((content - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        mx = (content + (m - 1.0) * 1e4).max(dim=1).values

        if want_rich:
            rich_out.append(torch.cat([nrm(mean), nrm(var.sqrt()), nrm(mx), ro],
                                      dim=-1).cpu())

        # §3.1's decomposition: the accumulation account predicts the register substate gains
        # even where the total loses, so the two must be estimated apart. Registers are also
        # emitted mean-pooled, which is dimension-matched to the single readout position and
        # so separates "the registers hold more" from "five vectors beat one".
        if n_reg:
            reg = h[:, -n_reg:, :]
            reg_out.append(nrm(reg.reshape(h.shape[0], -1)).cpu())
            regmean_out.append(nrm(reg.mean(dim=1)).cpu())
        else:
            z = torch.zeros(h.shape[0], 1, device=h.device)
            reg_out.append(z.cpu())
            regmean_out.append(z.cpu())
        cont_out.append(nrm(mean).cpu())

        print(f"\r  {desc}: {min(start + batch_size, len(items))}/{len(items)}",
              end="", flush=True)
    print()
    f64 = lambda xs: torch.cat(xs).numpy().astype(np.float64)
    return {"ro": f64(ro_out), "rich": f64(rich_out) if want_rich else None,
            "reg": f64(reg_out), "regmean": f64(regmean_out), "cont": f64(cont_out)}


# The contrast is fixed in advance rather than read off the data. Selecting the peak round on
# the same test split the interval is computed from is a winner's curse: the selected round is
# the one whose noise happened to be favourable, so the difference to it is biased away from
# zero. Round 2 is chosen because both families are flat through it in every run we have seen,
# not because it is any run's maximum.
CONTRAST = (1, -1)   # indices into rows: round 2 -> final round


def _ridge_mse(Xa, Ya, Xb, Yb, alpha, mu, sd, ymu):
    A = (Xa - mu) / sd
    B = (Xb - mu) / sd
    n, d = A.shape
    if d > n:  # dual form: cheaper, identical solution
        W = A.T @ np.linalg.solve(A @ A.T + alpha * np.eye(n), Ya - ymu)
    else:
        W = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ (Ya - ymu))
    return ((Yb - (B @ W + ymu)) ** 2).mean(0), (Yb - (B @ W + ymu)) ** 2


def select_alpha_across_rounds(Xs, Y, n_train, alphas):
    """One penalty per family, chosen on validation error summed over rounds.

    Selecting per round lets the estimator's own hyperparameter move with the quantity being
    estimated: a penalty that switches between rounds puts a step in the curve that belongs to
    the selection, not to the model. It cost us two spurious features already -- a jump in one
    checkpoint's mean-pooled registers, and a variance blow-up in the other's when the switch
    landed on a contrast endpoint. A shared penalty makes every across-round comparison
    penalty-matched, which is what a trend comparison needs.
    """
    n_val = max(1, int(0.25 * n_train))
    totals = []
    for a in alphas:
        s = 0.0
        for X in Xs:
            Xtr, Xval = X[:n_train - n_val], X[n_train - n_val:n_train]
            Ytr, Yval = Y[:n_train - n_val], Y[n_train - n_val:n_train]
            mu, sd, ymu = Xtr.mean(0), Xtr.std(0) + 1e-6, Ytr.mean(0)
            s += _ridge_mse(Xtr, Ytr, Xval, Yval, a, mu, sd, ymu)[0].mean()
        totals.append(s)
    return alphas[int(np.argmin(totals))]


def usable_information(X, Y, n_train, alphas, n_boot=200, rng=None):
    """Held-out usable information in nats, plus R^2 and a bootstrap SE.

    Both the model and the no-input baseline are fit on train and scored on test, so nothing
    about the test split enters either predictor. The baseline is the train mean of Y, which
    is the honest instantiation of the optional-ignorance predictor Assumption A requires;
    an earlier version estimated the baseline variance *on the test split*, which fits the
    noise scale to the evaluation data and biases the estimate upward.

    No per-dimension clipping. Clipping at zero discards the dimensions where the model does
    worse than the marginal, which is exactly the evidence against it, and makes the total
    depend on the coordinate system.
    """
    rng = rng or np.random.default_rng(0)
    Xtr_all, Xte = X[:n_train], X[n_train:]
    Ytr_all, Yte = Y[:n_train], Y[n_train:]
    best_alpha = alphas if np.isscalar(alphas) else select_alpha_across_rounds(
        [X], Y, n_train, alphas)

    mu, sd, ymu = Xtr_all.mean(0), Xtr_all.std(0) + 1e-6, Ytr_all.mean(0)
    mse, per_item = _ridge_mse(Xtr_all, Ytr_all, Xte, Yte, best_alpha, mu, sd, ymu)
    base_per_item = (Yte - ymu) ** 2          # baseline predictor, also fit on train only
    base = base_per_item.mean(0)

    nats = float(0.5 * np.log(base / np.maximum(mse, 1e-12)).sum())
    r2 = float(1.0 - mse.sum() / base.sum())

    idx = rng.integers(0, len(Yte), size=(n_boot, len(Yte)))
    boots = [0.5 * np.log(base_per_item[i].mean(0)
                          / np.maximum(per_item[i].mean(0), 1e-12)).sum() for i in idx]
    # per_item is returned so that comparisons *across rounds* can be bootstrapped on common
    # indices. Rounds share the test items, the target and the split, so the interval on a
    # difference is far tighter than the two level intervals suggest, and the level interval
    # is the wrong one to judge a trend by.
    return nats, r2, float(best_alpha), float(np.std(boots)), per_item, base_per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",
                    default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--image_dir", default="/home/zhoutuowen/data/MMEB-train")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--num_pairs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_t", type=int, default=6)
    ap.add_argument("--n_pca", type=int, default=64)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_args = DataArguments()
    queries, candidates = load_pairs(args.subset, args.num_pairs, args.image_dir, args.seed)
    print(f"{args.subset}: {len(queries)} pairs")

    # --- Y: counterpart content under a frozen encoder that never saw our training ---
    print("\n=== encoding Y = c(x+) with the frozen base model ===")
    base_args = ModelArguments(model_name=args.model_name, model_backbone=QWEN2_VL,
                               lora=False, pooling="eos", normalize=True)
    processor = load_processor(base_args, data_args)
    base = MMEBModel.load(base_args, is_trainable=False, processor=processor)
    base = base.to("cuda", dtype=torch.bfloat16).eval()
    Y = encode_states(base, processor, candidates, data_args, args.batch_size,
                      "Y", want_rich=False)["ro"]
    del base
    torch.cuda.empty_cache()

    # PCA keeps the per-dimension MSE estimates well conditioned at this sample size, and is
    # applied identically to every predictor family so it cannot favour any of them. It is
    # fit on the *training* pairs only: fitting it on all pairs, as an earlier version did,
    # leaks the test targets' covariance into the basis the test error is measured in.
    # Fit on the ridge *sub*-train only. Fitting on all 420 training pairs leaves the test
    # split clean but lets the basis see the validation slice that selects the penalty, so
    # the selection is not made on strictly held-out data.
    n_train = int(args.train_frac * len(queries))
    n_sub = n_train - max(1, int(0.25 * n_train))
    ymean = Y[:n_sub].mean(0)
    _, S, Vt = np.linalg.svd(Y[:n_sub] - ymean, full_matrices=False)
    n_pca = min(args.n_pca, Vt.shape[0])
    Yp = (Y - ymean) @ Vt[:n_pca].T
    # See the note in probe_intraloop_accumulation.py: fingerprinting the target makes "the
    # same Y across probes" checkable instead of inferred from two scripts naming one model.
    y_fp = hashlib.sha256(np.round(Y, 6).tobytes()).hexdigest()[:16]
    print(f"Y: {Y.shape} -> PCA {Yp.shape} (basis from {n_sub} sub-train pairs), "
          f"sub-train variance retained {float((S[:n_pca]**2).sum()/(S**2).sum()):.6f}, "
          f"Y fingerprint {y_fp}")

    # --- H^(t) from the trained checkpoint ---
    reloop_pt = os.path.join(args.checkpoint_path, "reloop.pt")
    state = (torch.load(reloop_pt, map_location="cpu", weights_only=False)
             if os.path.exists(reloop_pt)
             else {"reloop_t": 1, "reloop_m": 0,
                   "reloop_loop_start": 17, "reloop_loop_end": 27})
    print(f"\ncheckpoint topology: {state}")

    import glob as _glob
    is_fullft = (os.path.exists(os.path.join(args.checkpoint_path, "model.safetensors"))
                 or bool(_glob.glob(os.path.join(args.checkpoint_path, "model-*.safetensors"))))
    load_from = args.checkpoint_path if is_fullft else args.model_name

    model_args = ModelArguments(
        model_name=load_from, checkpoint_path=args.checkpoint_path,
        model_backbone=QWEN2_VL, lora=not is_fullft, pooling="eos", normalize=True,
        reloop_t=state["reloop_t"], reloop_m=state["reloop_m"],
        reloop_loop_start=state["reloop_loop_start"],
        reloop_loop_end=state["reloop_loop_end"])
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    try:
        sched, where = find_schedule(model)
    except RuntimeError:
        from src.model.reloop import attach_recurrence
        model.encoder.config.use_cache = False
        if hasattr(model.encoder.config, "text_config"):
            model.encoder.config.text_config.use_cache = False
        sched = attach_recurrence(model.encoder, state["reloop_loop_start"],
                                  state["reloop_loop_end"], 1)
        where = "manual"
    print(f"schedule at {where}: {sched}")

    # 1e-2..1e6 brackets every selection we have seen, so the optimum is interior rather than
    # pinned at a grid edge.
    alphas = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    families = [("ro", "u"), ("rich", "seq"), ("reg", "reg"),
                ("regmean", "regmean"), ("cont", "cont")]
    all_feats = []
    for t in range(1, args.max_t + 1):
        sched.num_loops = t
        print(f"\n--- encoding T={t} ---")
        all_feats.append(encode_states(model, processor, queries, data_args, args.batch_size,
                                       f"T={t} query", want_rich=True))

    rows = [{"T": t} for t in range(1, args.max_t + 1)]
    resid, baseline_resid, chosen = {}, {}, {}
    for key, tag in families:
        chosen[tag] = select_alpha_across_rounds([f[key] for f in all_feats], Yp,
                                                 n_train, alphas)
        for row, feats in zip(rows, all_feats):
            nats, r2, alpha, se, per_item, base_pi = usable_information(
                feats[key], Yp, n_train, chosen[tag])
            row.update({f"{tag}_nats": nats, f"{tag}_r2": r2,
                        f"{tag}_alpha": alpha, f"{tag}_se": se})
            resid.setdefault(tag, []).append(per_item)
            baseline_resid[tag] = base_pi
    # V_rich contains V_ro's columns, so the population optimum over the larger family cannot
    # be worse. A previous version took max(seq, u) of the *test* scores and called that the
    # nested estimate; picking a family by its test score is exactly the post-selection this
    # file is otherwise careful to avoid. Neither number is a ceiling estimate, so both are
    # reported and no max is taken.
    print("\nshared penalty per family: "
          + ", ".join(f"{t}={a:g}" for t, a in chosen.items()))
    for row in rows:
        print(f"  T={row['T']}  u={row['u_nats']:7.3f}±{row['u_se']:.3f} "
              f"(R2 {row['u_r2']:+.4f})  seq={row['seq_nats']:7.3f}  "
              f"reg={row['reg_nats']:7.3f}  regmean={row['regmean_nats']:7.3f}  "
              f"cont={row['cont_nats']:7.3f}")

    print(f"\n=== {args.subset} @ {os.path.basename(args.checkpoint_path)} "
          f"(trained T={state['reloop_t']}) ===")
    print(f"{'T':>2s} {'u(ro)':>9s} {'+-':>6s} {'u R2':>8s} {'seq':>9s} {'reg5':>9s} "
          f"{'regmean':>9s} {'cont':>8s} {'cont R2':>8s}")
    for r in rows:
        print(f"{r['T']:>2d} {r['u_nats']:>9.3f} {r['u_se']:>6.3f} {r['u_r2']:>+8.4f} "
              f"{r['seq_nats']:>9.3f} {r['reg_nats']:>9.3f} "
              f"{r['regmean_nats']:>9.3f} {r['cont_nats']:>8.3f} {r['cont_r2']:>+8.4f}")

    # Paired over test items: the same 180 pairs, target and split are used at every round,
    # so resampling them in common cancels the item-difficulty variance that dominates the
    # level intervals above.
    rng = np.random.default_rng(0)
    n_te = len(baseline_resid["u"])
    n_boot = 10000
    idx = rng.integers(0, n_te, size=(n_boot, n_te))
    a, b = CONTRAST
    paired = {}
    for tag in resid:
        base = baseline_resid[tag]
        nats_b = np.stack([[0.5 * np.log(base[i].mean(0)
                                         / np.maximum(pi[i].mean(0), 1e-12)).sum()
                            for i in idx] for pi in (resid[tag][a], resid[tag][b])])
        d = nats_b[1] - nats_b[0]
        n_ge = int((d >= 0).sum())
        paired[tag] = {
            "contrast": f"round {a+1} -> round {len(rows) if b == -1 else b+1}",
            "observed": rows[b][f"{tag}_nats"] - rows[a][f"{tag}_nats"],
            "boot_mean": float(np.mean(d)), "se": float(np.std(d)),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            # Monte Carlo resolution is 1/(n_boot+1); reporting anything smaller is reporting
            # the number of resamples, not the strength of the evidence.
            "n_boot": n_boot, "n_ge_0": n_ge,
            "p_ge_0": (n_ge + 1) / (n_boot + 1)}

    print(f"\npaired over test items, contrast fixed in advance (round {a+1} -> final), "
          f"{n_boot} resamples")
    for tag, p in paired.items():
        pstr = (f"<{1/(n_boot+1):.5f}" if p["n_ge_0"] == 0 else f"={p['p_ge_0']:.5f}")
        print(f"  {tag:>8s}  observed={p['observed']:>+7.3f}  boot={p['boot_mean']:>+7.3f}  "
              f"se={p['se']:.3f}  95% CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  "
              f"P(delta>=0){pstr}")

    # Residuals are cached so that re-analysis — a different contrast, a different interval —
    # costs no GPU. Every re-analysis in this paper so far has required re-encoding.
    np.savez_compressed(args.output.replace(".json", "_resid.npz"),
                        **{f"{tag}_r{i}": r for tag, rs in resid.items()
                           for i, r in enumerate(rs)},
                        **{f"{tag}_base": v for tag, v in baseline_resid.items()})

    b = rows[0]
    print("\nchange from T=1:")
    for r in rows[1:]:
        print(f"{r['T']:>2d}  du={r['u_nats']-b['u_nats']:>+8.3f}  "
              f"dreg={r['reg_nats']-b['reg_nats']:>+8.3f}  "
              f"dregmean={r['regmean_nats']-b['regmean_nats']:>+8.3f}")
    print("\nunder test: does u_t rise (the frame), and does the register substate rise "
          "while the rest falls (the accumulation account)?")
    print("read nats and R2 together: where they disagree, the effect is not robust.")

    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint_path, "subset": args.subset,
                   "trained_t": state["reloop_t"], "num_pairs": len(queries),
                   "n_train": n_train, "n_pca": n_pca, "y_fingerprint": y_fp,
                   "pca_fit_on": "train_only", "baseline": "train_mean",
                   "paired_peak_to_final": paired, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
