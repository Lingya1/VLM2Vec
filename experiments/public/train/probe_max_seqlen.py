"""扫出 20 个子集里真正的最长输入序列，为显存压力测试确定最坏情况。

背景：DataArguments.max_len 默认是 None，训练时不做任何截断。训练日志里看到的 1309 token
只是若干个 batch 内最大值的观测，不是全集上界。一个 45 小时的跑动会遍历 47 万条，只要
存在长尾样本迟早会撞上，而那一步的显存需求可能远高于平时。所以决定 GradCache 分块大小
之前，必须先知道最坏情况有多坏、且落在哪个子集上。

两个容易算错的地方，都踩过：

1. 视觉 token 的单位。Qwen2-VL 的 patch_size=14、merge_size=2，一个视觉 token 对应
   28x28 像素。resize_max_pixels 默认写成 `28*28*1280`，字面含义是 1280 个视觉 token，
   不是 1280 个 patch。
2. 不能假设每张图都顶到上限。上限只是 clamp，实际 token 数由图像自身分辨率决定。
   N24News/WebQA 的图偏小（实测整条序列只有 872 token），而 DocVQA/InfographicsVQA
   是高分辨率文档扫描件，才会真正顶到 1280。按"都顶满"估会挑错压测子集。

因此这里对每个子集抽样真实图像，用与 Qwen2VLImageProcessor 相同的 smart_resize 规则
算出实际视觉 token 数。只读图像头部拿尺寸，不解码像素，所以很快。

用法:
    cd /home/zhoutuowen/VLM2Vec
    PYTHONNOUSERSITE=1 PYTHONPATH=. HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python experiments/public/train/probe_max_seqlen.py
"""
import math
import os
import yaml
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer

Image.MAX_IMAGE_PIXELS = None

CONFIG = "experiments/public/train/train_image20_30k.yaml"
MODEL = "/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct"
IMAGE_ROOT = "/home/zhoutuowen/data/MMEB-train"

PATCH, MERGE = 14, 2
FACTOR = PATCH * MERGE                      # 28，即一个视觉 token 的边长（像素）
MIN_PIXELS = 28 * 28 * 4
MAX_PIXELS = 28 * 28 * 1280
MAX_VISUAL_TOKENS = MAX_PIXELS // (FACTOR * FACTOR)   # 1280

# 每个子集抽样的条数。文本按字符数取长尾，图像取分辨率最大的若干张。
SAMPLE = 400
TOP_N = 24


def visual_tokens(w, h):
    """复刻 Qwen2VLImageProcessor.smart_resize，返回该图展开后的视觉 token 数。"""
    h_bar = max(FACTOR, round(h / FACTOR) * FACTOR)
    w_bar = max(FACTOR, round(w / FACTOR) * FACTOR)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(h * w / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        h_bar = math.ceil(h * beta / FACTOR) * FACTOR
        w_bar = math.ceil(w * beta / FACTOR) * FACTOR
    return (h_bar // FACTOR) * (w_bar // FACTOR)


def image_tokens(path):
    if not path:
        return 0
    full = os.path.join(IMAGE_ROOT, path)
    try:
        with Image.open(full) as im:      # 只读头部，不解码
            return visual_tokens(*im.size)
    except Exception:
        return 0


def scan_side(ds, text_col, img_col, tok, n):
    """返回该侧最长的 (文本 token, 视觉 token, 总长)。"""
    texts = ds[text_col][:n]
    paths = ds[img_col][:n] if img_col in ds.column_names else [""] * len(texts)

    best = (0, 0, 0)
    # 文本长尾与图像大图未必落在同一条上，各取候选再合并评估
    by_text = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""), reverse=True)[:TOP_N]
    cand = set(by_text)
    sizes = [(i, image_tokens(paths[i])) for i in range(len(texts))]
    cand |= {i for i, _ in sorted(sizes, key=lambda t: t[1], reverse=True)[:TOP_N]}
    vis = dict(sizes)

    for i in cand:
        s = texts[i] or ""
        n_txt = len(tok(s.replace("<|image_1|>", ""), add_special_tokens=False)["input_ids"])
        total = n_txt + vis.get(i, 0)
        if total > best[2]:
            best = (n_txt, vis.get(i, 0), total)
    return best


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(MODEL)

    rows = []
    for name, sub in cfg.items():
        ds = load_dataset(sub["dataset_name"], sub["subset_name"], split=sub["dataset_split"])
        n = min(SAMPLE, ds.num_rows)
        q = scan_side(ds, "qry", "qry_image_path", tok, n)
        p = scan_side(ds, "pos_text", "pos_image_path", tok, n)
        rows.append((name, q, p))
        print(f"  扫完 {name:<18s} 查询 {q[2]:>5d}  目标 {p[2]:>5d}", flush=True)

    rows.sort(key=lambda r: max(r[1][2], r[2][2]), reverse=True)
    print(f"\n{'子集':<18s} | {'查询 文本+视觉=总长':>26s} | {'目标 文本+视觉=总长':>26s}")
    print("-" * 78)
    for name, q, p in rows:
        qs = f"{q[0]:>4d} + {q[1]:>4d} = {q[2]:>5d}"
        ps = f"{p[0]:>4d} + {p[1]:>4d} = {p[2]:>5d}"
        print(f"{name:<18s} | {qs:>26s} | {ps:>26s}")

    wq = max(rows, key=lambda r: r[1][2])
    wp = max(rows, key=lambda r: r[2][2])
    print(f"\n查询侧最长: {wq[0]}  {wq[1][2]} token")
    print(f"目标侧最长: {wp[0]}  {wp[2][2]} token")
    print(f"单图视觉 token 上限 {MAX_VISUAL_TOKENS}；抽样 {SAMPLE} 条/子集")
    print("压测应选查询与目标两侧都靠前的子集，因为一步里两侧都要过前向。")


if __name__ == "__main__":
    main()
