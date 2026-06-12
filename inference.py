"""
Inference script for CPET multi-panel Pulmonary Hypertension classification.

Runs a trained ConvNeXt model on a directory of preprocessed plot folders and
prints per-sample predictions and probabilities.

Usage:
    python inference.py --data_dir sample_data/ --checkpoint checkpoints/ConvNeXt_Baseline/fold_0/best_model.pth
    python inference.py --data_dir sample_data/ --checkpoint checkpoints/ConvNeXt_Baseline/fold_0/best_model.pth --model_type convnext
"""

import argparse
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model_chartclip import MultiPlotConvNeXt, MultiPlotResNet


PANEL_NAMES = [f"plot_{i:02d}.png" for i in range(1, 10)]

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        NORMALIZE,
    ])


def load_sample(sample_dir: Path, transform: T.Compose) -> torch.Tensor:
    panels = []
    for name in PANEL_NAMES:
        path = sample_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing panel: {path}")
        img = Image.open(path).convert("RGB")
        panels.append(transform(img))
    return torch.stack(panels, dim=0)  # [9, C, H, W]


def load_model(checkpoint_path: str, model_type: str, device: torch.device) -> torch.nn.Module:
    if model_type == "convnext":
        model = MultiPlotConvNeXt(
            convnext_model="facebook/convnext-base-224-22k",
            pretrained=False,
            freeze_encoder=False,
            projection_dim=None,
            num_panels=9,
            num_heads=8,
            num_transformer_layers=3,
            dropout=0.15,
            mlp_hidden_dim=256,
            mlp_ratio=4.0,
        )
    elif model_type == "resnet":
        model = MultiPlotResNet(
            resnet_model="resnet50",
            pretrained=False,
            freeze_encoder=False,
            projection_dim=None,
            num_panels=9,
            num_heads=8,
            num_transformer_layers=3,
            dropout=0.1,
            mlp_hidden_dim=256,
            mlp_ratio=4.0,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="CPET PH inference")
    parser.add_argument("--data_dir",   required=True, help="Directory containing sample sub-folders")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--model_type", default="convnext", choices=["convnext", "resnet"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--threshold",  type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--output_csv", default=None, help="Optional path to save predictions CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model_type, device)
    transform = get_transform(args.image_size)

    data_dir = Path(args.data_dir)
    sample_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        print(f"No sub-directories found in {data_dir}")
        return

    print(f"\n{'Sample':<20} {'Prob (PH)':<12} {'Prediction'}")
    print("-" * 45)

    results = []
    with torch.no_grad():
        for sample_dir in sample_dirs:
            try:
                panels = load_sample(sample_dir, transform).unsqueeze(0).to(device)  # [1, 9, C, H, W]
                logit = model(panels)
                prob = torch.sigmoid(logit).item()
                pred = "PH" if prob >= args.threshold else "Normal"
                print(f"{sample_dir.name:<20} {prob:<12.4f} {pred}")
                results.append({"sample": sample_dir.name, "prob_ph": round(prob, 4), "prediction": pred})
            except FileNotFoundError as e:
                print(f"{sample_dir.name:<20} ERROR: {e}")

    if args.output_csv:
        out = Path(args.output_csv)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample", "prob_ph", "prediction"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPredictions saved to {out}")


if __name__ == "__main__":
    main()
