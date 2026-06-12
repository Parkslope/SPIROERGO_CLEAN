# Deep Learning on Wassermann Plots for Pulmonary Hypertension Detection and Subtype Discrimination

Code for the paper:  
**"Deep Learning on Wassermann Plots for Pulmonary Hypertension Detection and Subtype Discrimination"**  
Tri-Thien Nguyen, Andreas Maier, Michael Uder, Sebastian Bickelhaupt, Julian Mueller, Simon R. Schneider, Michael Furian, Helga Preiss, Carmen Wick, Silvia Ulrich, Mona Lichtblau  
*[Journal Name], [Year]*

Trained model weights: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20667420.svg)](https://doi.org/10.5281/zenodo.20667420)

---

## Overview

This repository implements a deep learning system for automated detection of Pulmonary Hypertension (PH) from Cardiopulmonary Exercise Testing (CPET) multi-panel plots (Wasserman plots). A shared CNN encoder processes each of 9 panels independently, and a Transformer aggregator captures inter-panel relationships for binary classification.

### Key features

- Multi-encoder support: ResNet-50, ConvNeXt-Base, CLIP ViT-L/14
- Transformer aggregation with learnable positional embeddings for 9 panels
- 5-fold cross-validation with nested validation splits
- Panel-shuffling ablation to assess positional dependence
- Attention weight visualization (panel-level heatmaps)
- GradCAM subgroup analysis (PAH vs. CTEPH)
- 20 anonymized sample cases included

## Architecture

```
Input: 9 CPET panels (3×3 grid)
    ↓
Shared encoder (ResNet-50 / ConvNeXt-Base / CLIP ViT-L)
    ↓
Learnable position embeddings (9 positions)
    ↓
Transformer aggregator (3 layers, 8 heads) + CLS token
    ↓
MLP classification head
    ↓
Output: P(Pulmonary Hypertension)
```

## Quick start

### Installation

```bash
pip install -r requirements.txt
```

### Inference on sample data

Download the trained ConvNeXt checkpoint from Zenodo (see link above), then:

```bash
python inference.py \
    --data_dir sample_data/ \
    --checkpoint checkpoints/ConvNeXt_Baseline/fold_0/best_model.pth
```

### Preprocess your own CPET PDFs

```bash
python preprocessing/preprocess_plots.py \
    --input_dir /path/to/cpet_pdfs/ \
    --output_dir /path/to/preprocessed_plots/
```

This renders each PDF, detects the 3×3 plot grid via bounding-box analysis, and saves 9 panel PNGs per case.

### Prepare cross-validation splits

```bash
python preprocessing/create_nested_folds.py \
    --csv your_labels.csv \
    --n_folds 5
```

### Train (5-fold cross-validation)

```bash
# ConvNeXt-Base (best model)
python train_cv.py \
    --config configs/config_convnext.yaml \
    --model_name ConvNeXt_Baseline \
    --csv your_labels.csv \
    --plot_dir /path/to/preprocessed_plots/

# Panel-shuffling ablation
python train_cv.py \
    --config configs/config_convnext.yaml \
    --model_name ConvNeXt_ShuffledPanels \
    --csv your_labels.csv \
    --plot_dir /path/to/preprocessed_plots/ \
    --shuffle_panels

# ResNet-50 baseline
python train_cv.py \
    --config configs/config_resnet50.yaml \
    --model_name ResNet50_Baseline \
    --csv your_labels.csv \
    --plot_dir /path/to/preprocessed_plots/
```

### Evaluate

```bash
# Aggregate CV metrics
python evaluation/evaluate_cv_results.py \
    --results_dir results/ConvNeXt_Baseline

# Statistical model comparison (DeLong test)
python evaluation/model_comparison_delong.py

# Panel attention heatmaps
python evaluation/analyze_attention_weights.py \
    --plot_dir /path/to/preprocessed_plots/ \
    --checkpoint_dir checkpoints/ConvNeXt_Baseline

# PAH vs. CTEPH subgroup attention
python evaluation/analyze_attention_weights_subgroup.py \
    --plot_dir /path/to/preprocessed_plots/ \
    --csv_file your_subgroup_labels.csv

# PAH vs. CTEPH GradCAM
python evaluation/analyze_attention_gradcam_subgroup.py \
    --plot_dir /path/to/preprocessed_plots/ \
    --csv_file your_subgroup_labels.csv
```

## Repository structure

```
├── inference.py                        # Inference on new samples
├── train_cv.py                         # 5-fold cross-validation training
├── train_chartclip.py                  # Single-run training (used by train_cv.py)
├── training.py                         # Training loop, metrics, early stopping
├── model_chartclip.py                  # Model architectures
├── data_loader_multiplot.py            # Dataset and data loaders
├── requirements.txt
│
├── configs/                            # Hydra configuration files
│   ├── config_convnext.yaml            # ConvNeXt-Base (best)
│   ├── config_resnet50.yaml            # ResNet-50 baseline
│   ├── config.yaml                     # CLIP ViT-L baseline
│   ├── model/
│   ├── data/
│   ├── training/
│   └── augmentation/
│
├── preprocessing/
│   ├── preprocess_plots.py             # PDF → 9 panel PNGs
│   ├── create_nested_folds.py          # 5-fold CV split creation
│   └── check_class_balance.py
│
├── evaluation/
│   ├── evaluate_cv_results.py          # Aggregate fold metrics + ROC curves
│   ├── model_comparison_delong.py      # DeLong AUC comparison (all samples)
│   ├── model_comparison_delong_subgroup.py  # DeLong AUC comparison (subgroup)
│   ├── analyze_attention_weights.py    # Panel attention heatmaps (PH vs non-PH)
│   ├── analyze_attention_weights_subgroup.py  # Attention heatmaps (PAH vs CTEPH)
│   ├── analyze_attention_gradcam_subgroup.py  # GradCAM (PAH vs CTEPH)
│   └── create_combined_figures.py      # Publication figures
│
├── utils/
│   ├── __init__.py
│   └── visualization.py
│
└── sample_data/                        # 20 anonymized CPET cases
    ├── sample_labels.csv
    ├── sample_001/
    │   ├── plot_01.png ... plot_09.png
    └── ...
```

## Dataset

- **Total samples**: 2,125 anonymized CPET multi-panel plots
- **Classes**: Binary — 0 = Normal/Non-PH, 1 = Pulmonary Hypertension
- **Splits**: 5-fold cross-validation with nested validation folds
- **Sample data**: 20 representative de-identified cases in `sample_data/`

The CSV label file requires columns: `sample_id`, `label`, `fold_test`, `fold_val`.

## Results

| Model | AUC (mean ± std) |
|-------|-----------------|
| ConvNeXt-Base | **0.846 ± 0.020** |
| ConvNeXt-Base (shuffled panels) | 0.848 ± 0.027 |
| ResNet-50 | 0.831 ± 0.022 |
| CLIP ViT-L/14 | 0.797 ± 0.022 |

The panel-shuffling ablation shows no significant performance drop, indicating the transformer learns position-invariant inter-panel features rather than relying on fixed spatial ordering.

## Pretrained models

Trained model weights (5-fold checkpoints for ConvNeXt-Baseline and ConvNeXt-ShuffledPanels) are available on Zenodo:

**DOI: [https://doi.org/10.5281/zenodo.20667420](https://doi.org/10.5281/zenodo.20667420)**

Expected structure after download:

```
checkpoints/
├── ConvNeXt_Baseline/
│   ├── fold_0/best_model.pth
│   ├── fold_1/best_model.pth
│   ├── fold_2/best_model.pth
│   ├── fold_3/best_model.pth
│   └── fold_4/best_model.pth
└── ConvNeXt_ShuffledPanels/
    ├── fold_0/best_model.pth
    └── ...
```

## Configuration

Key hyperparameters (see `configs/` for full settings):

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 (ConvNeXt/ResNet), 336×336 (CLIP) |
| Batch size | 8–16 |
| Transformer layers | 3 |
| Attention heads | 8 |
| MLP ratio | 4.0 |
| Dropout | 0.15 |
| Optimizer | AdamW, lr=3×10⁻⁴ |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Early stopping | patience=15, monitor=val_AUC |
| Label smoothing | 0.1 |

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA-capable GPU recommended (training); inference runs on CPU

See `requirements.txt` for the full list.

## Citation

If you use this code, please cite:

```bibtex
@article{nguyen2025spiroergo,
  title   = {Deep Learning on Wassermann Plots for Pulmonary Hypertension Detection and Subtype Discrimination},
  author  = {Nguyen, Tri-Thien and Maier, Andreas and Uder, Michael and Bickelhaupt, Sebastian and Mueller, Julian and Schneider, Simon R. and Furian, Michael and Preiss, Helga and Wick, Carmen and Ulrich, Silvia and Lichtblau, Mona},
  journal = {[Journal Name]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Ethics and data availability

All CPET recordings were de-identified prior to analysis. The 20 sample cases provided in this repository contain no Protected Health Information (PHI). The full dataset cannot be shared publicly due to patient privacy regulations; access may be requested from the corresponding author subject to a data use agreement.
