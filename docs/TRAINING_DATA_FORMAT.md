# VLM2Vec 训练数据格式说明

本文档说明 VLM2Vec 训练 pipeline 期望的数据格式。

## 一、核心概念

VLM2Vec 采用 **对比学习 / 检索对** 训练：每个样本由 **query (qry)** 和 **positive target (pos)** 组成，模型学习让 query 和 positive 的嵌入在向量空间中靠近。

---

## 二、最终 Collator 期望的格式

经过 `dataset.map()` 处理后，每个 batch 必须包含以下字段（见 `base_pair_dataset.py` 的 `MULTIMODAL_FEATURES`）：

```python
{
    "query_text": str,      # 查询文本（含图像占位符，如 <|image_pad|> 或 <image>）
    "query_image": {        # 查询图像（可为 None 表示纯文本查询）
        "paths": [str],     # 图像路径列表
        "bytes": [bytes],   # 或预加载的 bytes
        "resolutions": [[w, h], ...]  # 每张图的 [宽, 高]
    },
    "pos_text": str,        # 正例文本
    "pos_image": {...},     # 正例图像（可为空占位）
    "neg_text": [str],     # 负例文本列表（可选）
    "neg_image": [{...}],   # 负例图像列表（可选）
    "global_dataset_name": str  # 数据集名称（自动添加）
}
```

---

## 三、MMEB 数据源格式（dataset_parser: mmeb）

从 HuggingFace `TIGER-Lab/MMEB-train` 加载时，**原始列**必须为：

| 列名 | 类型 | 说明 |
|------|------|------|
| `qry` | str | 查询文本（含图像占位符 `<\|image_1\|>`，会被替换为当前 backbone 的 token） |
| `qry_image_path` | str | 查询图像路径（相对 `image_dir` 的路径，可为空） |
| `pos_text` | str | 正例文本 |
| `pos_image_path` | str | 正例图像路径（可为空） |
| `neg_text` | str | 负例文本（可选） |
| `neg_image_path` | str | 负例图像路径（可选） |

**示例（OK-VQA 类型）：**

```json
{
  "qry": "<|image_1|> Represent the given image with the following question: What is the name of that toy in the background?",
  "qry_image_path": "MMEB-train/images/OK-VQA/xxx.jpg",
  "pos_text": "ring",
  "pos_image_path": "MMEB-train/images/OK-VQA/xxx.jpg",
  "neg_text": "",
  "neg_image_path": null
}
```

**图像占位符**：MMEB 默认使用 Phi3 的 `<|image_1|>`，训练时会自动替换为当前 backbone 的 token：
- Qwen2-VL / Qwen2.5-VL: `<|image_pad|>`
- LLaVA-Next: `<image>`

---

## 四、配置文件格式（YAML）

在 `experiments/public/train/train_image.yaml` 中：

```yaml
OK-VQA:
  dataset_parser: mmeb
  dataset_name: TIGER-Lab/MMEB-train
  subset_name: OK-VQA
  dataset_split: original
  image_dir: vlm2vec_train/MMEB-train/image   # 图像根目录
  num_sample_per_subset: 10000
  weight: 1
```

---

## 五、与 UME-sft-train 的差异

| 项目 | UME-sft-train | VLM2Vec (MMEB) |
|------|---------------|----------------|
| 结构 | `qry`/`pos` 嵌套 `conversations`、`image`、`video` | 扁平结构：`qry`、`qry_image_path`、`pos_text`、`pos_image_path` |
| 文本 | 对话格式，含 `disc_emb`、`gen_emb` 等 | 直接指令 + 文本/图像 |
| 图像 | 路径在 `qry.image`、`pos.image` | 路径在 `qry_image_path`、`pos_image_path` |
| 数据源 | 本地 JSON 大文件 | HuggingFace datasets |

**若要从 UME-sft-train 转换到 VLM2Vec**，需要：

1. 从 `qry.conversations[0].value` 提取查询文本（去掉 `<disc_emb>` 及后续指令）
2. 从 `pos.conversations[0].value` 提取正例文本（去掉 `<disc_emb>` 等）
3. 将 `qry.image`、`pos.image` 映射为 `qry_image_path`、`pos_image_path`
4. 图像占位符替换为 `<|image_1|>`（或对应 backbone 的 token）

---

## 六、其他 Parser 示例

### llavahound_qa（视频 QA）

```python
# 原始格式
{
  "id": "...",
  "video": "video_id",
  "conversations": [
    {"from": "human", "value": "<video>...question..."},
    {"from": "gpt", "value": "answer"}
  ]
}
# 需要 video_frame_basedir, num_frames 等参数
```

### vidore（文档/视频）

支持 `bytes` 形式的图像输入，无需本地路径。

---

## 七、数据流示意

```
HuggingFace 或 JSON
    ↓
dataset_parser (mmeb / llavahound_qa / ...)
    ↓
data_prepare_* 函数 → 输出 MULTIMODAL_FEATURES 格式
    ↓
MultimodalDataCollator
    ↓
process_vlm_inputs_fns → input_ids, pixel_values, ...
    ↓
model(qry=..., tgt=...) → 对比学习 loss
```

---

## 八、最小示例（JSON 格式）

若使用自定义 JSON 数据并写一个 mmeb 风格的 parser：

```json
[
  {
    "qry": "<|image_1|> Represent the given image with the following question: What color is the car?",
    "qry_image_path": "images/okvqa/001.jpg",
    "pos_text": "red",
    "pos_image_path": "images/okvqa/001.jpg"
  }
]
```

配合 `image_dir` 指向包含 `images/` 的根目录即可。
