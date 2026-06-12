# Evaluation Scripts

## evaluate_cv_results.py
Aggregates 5-fold CV metrics (AUC, accuracy, sensitivity, specificity, PPV, NPV, F1) and produces ROC curves with 95% CI.

```bash
python evaluation/evaluate_cv_results.py --results_dir results/ConvNeXt_Baseline
```

## model_comparison_delong.py
Pairwise DeLong AUC comparison across all models with Holm-Bonferroni correction.

```bash
python evaluation/model_comparison_delong.py
```

## model_comparison_delong_subgroup.py
Same as above but restricted to the PAH vs. CTEPH subgroup.

## analyze_attention_weights.py
Extracts CLS-to-panel attention weights from the Transformer aggregator and generates 3×3 heatmaps comparing PH vs. non-PH cases.

```bash
python evaluation/analyze_attention_weights.py \
    --plot_dir /path/to/preprocessed_plots/ \
    --checkpoint_dir checkpoints/ConvNeXt_Baseline
```

## analyze_attention_weights_subgroup.py
Panel attention heatmaps for PAH vs. CTEPH subgroup analysis.

## analyze_attention_gradcam_subgroup.py
GradCAM visualizations per panel for PAH vs. CTEPH cases.

## create_combined_figures.py
Generates publication-ready figures (AUC comparison bar chart, attention heatmap panels).
