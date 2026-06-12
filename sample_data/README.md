# Sample Data

This directory contains 20 anonymized representative samples from the dataset for testing and validation purposes.

## Structure

Each sample directory contains:
- `plot_01.png` through `plot_09.png`: Nine CPET plot panels
- Sample directories named `sample_001` through `sample_020`

## Labels

The `sample_labels.csv` file contains:
- `sample_id`: Anonymized sample identifier
- `label`: 0 = Normal, 1 = Pulmonary Hypertension
- `fold_test`: Test fold assignment (0-4)
- `fold_val`: Validation fold assignment (0-4)

## Usage

```python
import pandas as pd
from PIL import Image

# Load labels
labels = pd.read_csv('sample_data/sample_labels.csv')

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
