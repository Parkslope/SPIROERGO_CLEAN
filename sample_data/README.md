# Sample Data

This directory contains 20 anonymized representative samples from the dataset for testing and validation purposes.

## Structure

Each sample directory contains:
- `plot_01.png` through `plot_09.png`: Nine CPET plot panels
- Sample directories named `sample_001` through `sample_020`

## Labels

The `sample_labels.csv` file is semicolon-separated (same format as `data/csv_template.csv`) and contains:
- `id`: Row index
- `filename`: Sample folder name (e.g. `sample_001`)
- `binary_class`: 0 = Normal, 1 = Pulmonary Hypertension
- `split`: `train`, `val`, or `test` (fold 0 assignment, for single-run training)
- `fold_test`: Test fold assignment (0-4)
- `fold_val`: Validation fold assignment (0-4)

The fold assignments are placeholders for testing the pipeline, not the study folds.

## Usage

```python
import pandas as pd
from PIL import Image

# Load labels
labels = pd.read_csv('sample_data/sample_labels.csv', sep=';')

# Load a sample's plots
sample_id = 'sample_001'
plots = []
for i in range(1, 10):
    img = Image.open(f'sample_data/{sample_id}/plot_{i:02d}.png')
    plots.append(img)
```

## Anonymization

- All patient identifiers have been removed
- Files use anonymous sample IDs
- No Protected Health Information (PHI) included
- Representative subset of the full dataset
