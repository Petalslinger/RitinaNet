# RetinaNet 目标检测实验：MDA 注意力 + Soft-NMS

基于 `torchvision` 的 **RetinaNet-ResNet50-FPN-v2** 实现的目标检测实验工程：在 FPN 多尺度特征上接入**多维注意力（MDA）**，并用 **Soft-NMS** 替换检测头后的标准 NMS。在 **HRSC2016** 遥感舰船数据集与 **CCTSDB2021** 交通标志数据集上完成了基线对比、注意力消融、后处理消融与跨域迁移实验。

## 特性

- **严格对照的消融实验**：训练侧 `--use-mda / --no-mda`、评估侧 `--soft-nms`，四组实验（A/B/C/D）共享同一数据划分与超参。
- **MDA 多维注意力**：通道注意力与空间注意力并联，输出经可学习门控系数相加（初值 0），保证训练起点等价于原始 RetinaNet，不会破坏 COCO 预训练权重的特征分布。
- **Soft-NMS**：以高斯衰减替代硬阈值抑制，缓解密集小目标互相压制的漏检问题。
- **完整的实验链路**：数据下载 / 转换 / 自检 / 训练 / 评估 / 推理 / 绘图 / 结果汇总，全部为可复现的命令行脚本。
- **工程细节**：AMP 混合精度、梯度裁剪、断点续训（含 RNG 状态还原）、学习率调度、逐 epoch 指标日志。

## 目录结构

```
.
├── configs/                     # 三套实验配置（HRSC2016 / CCTSDB / 密集子集）
│   ├── hrsc2016_retinanet.yaml
│   ├── cctsdb_retinanet.yaml
│   └── cctsdb_dense_retinanet.yaml
├── src/
│   ├── model.py                 # RetinaNet + MDA 组装，挂载 Soft-NMS 后处理
│   ├── attention.py             # 通道 / 空间 / 多维注意力模块
│   ├── soft_nms.py              # Soft-NMS 实现
│   ├── dataset.py               # YOLO 格式数据集与 DataLoader
│   ├── prepare_data.py          # VOC / polygon 标注 → YOLO，划分 train/val/test
│   ├── download_data.py         # CCTSDB2021 下载、MD5 校验、解压
│   ├── select_dense.py          # 按每图目标数抽取密集子集
│   ├── train.py                 # 训练、验证、断点续训、指标日志
│   ├── evaluate.py              # mAP / P / R / F1 / FPS / 混淆矩阵
│   ├── predict.py               # 单图或目录推理并绘制检测框
│   ├── visualize.py             # 训练曲线、消融柱状图、混淆矩阵
│   ├── compare.py               # 多份评估 JSON → Markdown 对比表
│   ├── check_env.py             # 环境自检（Python / CUDA / 依赖 / 显存 / 数据）
│   └── check_dataset.py         # 数据集完整性自检
├── requirements.txt
└── .gitignore
```

## 环境

- Python >= 3.10，PyTorch >= 2.0，torchvision >= 0.15
- 单卡 8 GB 显存即可（默认 512/640 输入，batch size 2 + AMP）

```bash
conda create -n retinanet python=3.11 -y
conda activate retinanet

# 先按 https://pytorch.org 的说明安装匹配 CUDA 的 torch / torchvision，再装其余依赖
pip install -r requirements.txt

# 自检：Python、CUDA、依赖版本、显存、数据集状态
python -m src.check_env
```

> 显存不足时，把配置文件里的 `min_size / max_size` 降到 448/576，或把 `batch_size` 调成 1。

## 数据准备

仓库**不包含数据集**（体积与授权原因）。所有脚本约定数据根目录为 `data/processed/<name>`，采用 YOLO 格式：

```
data/processed/cctsdb/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── classes.txt
```

### CCTSDB2021（自动下载）

```bash
# 下载 1.5 GB 的 dataset.rar、校验 MD5、解压并转换为 YOLO 格式
python -m src.download_data --prepare

# 检查图片 / 标注 / 统计文件是否一致
python -m src.check_dataset --root data/processed/cctsdb
```

### HRSC2016（手动下载）

HRSC2016 需在官网申请后下载，解压得到包含 `Train/`、`Test/`、`ImageSets/` 的根目录，然后转换：

```bash
python -m src.prepare_data --raw-dir /path/to/HRSC2016 \
    --out-dir data/processed/hrsc2016 \
    --classes ship
```

脚本会自动识别 `<robndbox>` / `<HRSC_Object>` 旋转标注，取水平外接矩形。

### 密集子集（Soft-NMS 加分实验）

```bash
python -m src.select_dense --root data/processed/cctsdb \
    --out data/processed/cctsdb_dense3 --min-boxes 3
```

## 训练

```bash
# CCTSDB2021 基线（关闭 MDA）
python -m src.train --config configs/cctsdb_retinanet.yaml \
    --output outputs/checkpoints/cctsdb_base --no-mda

# CCTSDB2021 + MDA
python -m src.train --config configs/cctsdb_retinanet.yaml \
    --output outputs/checkpoints/cctsdb_mda --use-mda

# HRSC2016
python -m src.train --config configs/hrsc2016_retinanet.yaml \
    --output outputs/checkpoints/hrsc2016_mda --use-mda

# 断点续训
python -m src.train --config configs/hrsc2016_retinanet.yaml \
    --output outputs/checkpoints/hrsc2016_mda --resume outputs/checkpoints/hrsc2016_mda/last.pth
```

命令行参数会覆盖 YAML 配置；未显式给出的项才用配置文件的值。训练产物：`best.pth`、`last.pth`、`args.json`、`train_log.csv`。

## 评估与推理

```bash
# 测试集评估（可重复跑 A/B/C/D 四组）
python -m src.evaluate --checkpoint outputs/checkpoints/cctsdb_mda/best.pth \
    --data data/processed/cctsdb --split test --soft-nms \
    --save-pr outputs/figures/pr_cctsdb.png --confusion \
    --output outputs/metrics/cctsdb_mda_softnms.json

# 单图 / 目录推理
python -m src.predict --checkpoint outputs/checkpoints/cctsdb_mda/best.pth \
    --source path/to/images --output outputs/predictions --soft-nms
```

## 可视化与结果汇总

```bash
# 训练曲线
python -m src.visualize curves --log outputs/checkpoints/cctsdb_mda/train_log.csv \
    --output outputs/figures/curves.png

# 消融柱状图
python -m src.visualize ablation \
    --results outputs/metrics/A.json outputs/metrics/B.json \
              outputs/metrics/C.json outputs/metrics/D.json \
    --labels "基线+NMS" "基线+Soft-NMS" "MDA+NMS" "MDA+Soft-NMS" \
    --metric mAP50 --output outputs/figures/ablation.png

# 多份评估结果 → Markdown 对比表（--group 可重复，格式为 "标题|文件1,文件2,..."）
python -m src.compare \
    --group "HRSC2016|outputs/metrics/A.json,outputs/metrics/B.json,outputs/metrics/C.json,outputs/metrics/D.json" \
    --group "CCTSDB2021|outputs/metrics/cctsdb_A.json,outputs/metrics/cctsdb_B.json,outputs/metrics/cctsdb_C.json,outputs/metrics/cctsdb_D.json" \
    --output report/compare_tables.md
```

## 实验结果（mAP@0.5 / mAP@0.5:0.95）

**HRSC2016**（测试集 438 图 / 1228 实例）

| 组 | 实验 | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 基线 + NMS | 85.28% | 66.40% | 85.70% | 76.14% | 80.64% |
| B | 基线 + Soft-NMS | 85.79% | 67.50% | 84.09% | 80.05% | 82.02% |
| C | MDA + NMS | 85.96% | 66.60% | 86.51% | 76.79% | 81.36% |
| D | MDA + Soft-NMS | **86.65%** | **67.90%** | 85.07% | **80.29%** | **82.61%** |

**CCTSDB2021**（交通标志迁移实验）

| 组 | 实验 | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 基线 + NMS | 91.19% | 61.31% | 89.25% | 87.81% | 88.52% |
| B | 基线 + Soft-NMS | 92.10% | 63.22% | 91.27% | 88.31% | 89.76% |
| C | MDA + NMS | 91.93% | 61.58% | 87.57% | 89.74% | 88.64% |
| D | MDA + Soft-NMS | **92.40%** | **63.36%** | **91.56%** | 87.92% | 89.70% |

结论：MDA 与 Soft-NMS 收益互补，二者叠加在 HRSC2016 上带来 +1.37 个百分点的 mAP@0.5 提升；Soft-NMS 的主要贡献在召回率（76.14% → 80.05%）。迁移到交通标志场景后增益仍然存在，但幅度小于教材中的 small-DOTA 实验，说明注意力模块的收益与目标密度相关。

## 关于本仓库

- `data/`、`outputs/`、`*.pth`、`*.rar` 等数据与训练产物均在 `.gitignore` 中，仓库只保留代码与配置。
- 数据集版权归原作者所有，请遵守 HRSC2016 与 CCTSDB2021 的使用协议。
- 预训练权重来自 `torchvision` 的 COCO RetinaNet-ResNet50-FPN-v2。
