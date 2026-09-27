# MAG-Net: An interpretable gated multi-task network for provably bounded graph-based property analysis

_(Source code and experiment pipeline for the manuscript submitted to **Pattern Analysis and Applications**)_

**Status:** submitted to _Pattern Analysis and Applications_

This repository contains the full experimental pipeline for reproducibility of the paper:

> **MAG-Net: An interpretable gated multi-task network for provably bounded graph-based property analysis**
> Authors: Irfan Haider et al.

## Repository Layout

```
1-Topological Descriptor Computation/
├── topological_formulas.py                     # to display the LaTeX definitions of the descriptors
├── topological_descriptors_computation.py      # to reproduce the descriptor correlation matrix
└── CLI_run_commands.txt                        # guide to run the descriptor and correlation scripts

2-MAG-Net Trainings/
├── proposed_validated_pipeline.py              # to run the full validated MAG-Net pipeline
├── magnet_joint_tuning.py                      # to perform the joint hyperparameter search
├── magnet_tuned_kfold_summary.py               # to regenerate the per-property 5-fold CV summary
├── extract_gating_and_errors.py                # to extract W_g and per-object percentage errors
└── CLI_run_commands.txt                        # guide to run the trainings and figures

3-Baselines/
├── baseline_harness.py                        # shared harness for all baselines
├── classical_baselines.py                     # to train the classical ensemble baselines
├── baseline_smiles_cnn.py                     # to train the initial representation baseline
├── optimize_smiles_cnn.py                     # to perform the matched representation search
├── optimize_transformer_smiles.py             # to perform the Transformer baseline search
├── fair_tune_descriptor_mlp.py                # to perform the fair matched ungated ablation tuning
├── optimize_morgan_fp.py                      # to perform the fingerprint baseline search
└── CLI_run_commands.txt                       # guide to run the baselines

4-Dataset.txt                                  # dataset source and description
README.md
USAGE_NOTICE.txt                               # usage policy and citation instructions
requirements (MAG-Net).txt                     # MAG-Net environment dependencies
requirements (Baselines).txt                   # baselines environment dependencies
```

---

## Dataset

See `4-Dataset.txt` for detailed notes.

### MAG-Net Dataset Folder Structure

For example:

```
mag_net_dataset/
├── compound_structures.csv
│       # object identifiers and structural representations
├── structural_indices.csv
│       # graph-derived structural descriptors per object
├── target_properties.csv
│       # target property values per object
└── held_out_split.csv
        # fixed held-out evaluation split
```

### File Structure Schema

For example:

```
compound_structures.csv
├── identifier              # object identifier
├── representation          # structural representation string
└── group                   # optional grouping label

structural_indices.csv
├── identifier              # object identifier
├── index_01
├── index_02
├── index_03
├── index_04
├── index_05
├── index_06
├── index_07
├── index_08
└── index_09

target_properties.csv
├── identifier              # object identifier
├── target_01
├── target_02
├── target_03
├── target_04
├── target_05
├── target_06
└── target_07

held_out_split.csv
├── identifier
├── representation
└── group
```

### Partition Protocol

For example:

```
benchmark
├── training partition      # retained for model fitting
└── held-out partition      # reserved for final evaluation

split configuration
├── random_state            42
├── test_size               0.2
├── shuffle                 True
└── overlap control         no identifier or representation overlap
```

---

## Environment and Dependencies

This project uses **separate Conda environments** for framework-specific dependencies.

Install all requirements:

```
pip install -r requirements.txt
```

Main packages:

```
# MAG-Net environment
torch>=1.13
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
scipy>=1.9
matplotlib>=3.6
rdkit>=2024.09

# Baselines environment
torch>=1.13
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
scipy>=1.9
rdkit>=2024.09
matplotlib>=3.6
```

---

## Reproducing the Experiments

All experiments read from the `01_data/` directory and use the same
canonical split (seed = 42) throughout.

**1. Compute the degree-based descriptors**
* Run `topological_descriptors_computation.py` to build the topological descriptor table.
* Execute `topological_formulas.py` to display the closed-form LaTeX definitions of the descriptors.

**2. Reproduce the descriptor–property correlation analysis**
* Run `compute_descriptor_property_correlations.py` to recompute the Pearson correlation matrix between the descriptors and the seven target properties.

**3. Train and tune MAG-Net**
* Run `proposed_validated_pipeline.py` for the full validated training pipeline (5-fold cross-validation, Y-scrambling, applicability-domain check, and held-out test evaluation).
* Execute `magnet_joint_tuning.py` to perform the joint hyperparameter search under matched 5-fold cross-validation.

**4. Extract learned attribution and cross-validation summary**
* Run `extract_gating_and_errors.py` to extract the property-specific gating matrix W_g and per-object percentage errors.
* Execute `magnet_tuned_kfold_summary.py` to regenerate the per-property 5-fold cross-validation summary for the adopted configuration.

**5. Assess robustness**
* Run `repeated_split_eval_neural.py` to evaluate MAG-Net and the ungated ablation across independent random train/test splits.
* Execute `c5_b4_combined.py` to assess multi-seed training stability and input-perturbation robustness.

**6. Train the baseline models**
* Run `classical_baselines.py` to train the classical ensemble baselines via randomized search.
* Execute `baseline_smiles_cnn.py` to train the initial representation baseline.
* Run `optimize_smiles_cnn.py` for the representation baseline under matched search budget.
* Execute `optimize_transformer_smiles.py` for the Transformer baseline.
* Run `fair_tune_descriptor_mlp.py` for the ungated ablation baseline under a matched search.
* Execute `optimize_morgan_fp.py` for the fingerprint baseline.

**7. Assemble the primary benchmark**
* Combine the per-property metrics produced by Steps 3 and 6 to reproduce the primary benchmark tables reported in the paper.

**8. Regenerate the paper figures**
* Run `generate_attribution_heatmap.py` for the learned gating heatmap.
* Execute `generate_parity_merged.py` for the parity plots.
* Run `generate_violin_composite.py` for the split-violin distributions.
* Execute `regenerate_radar_v2.py` for the radar comparison across all models.

Each script includes parameters and instructions to reproduce the reported results in the paper.


---

## Citation

Please read `USAGE_NOTICE.txt` for legal terms.
Once the paper is published,:
Citation (upon publication)
```
```

---

## License & Usage Policy

Copyright © 2026 Irfan Haider.

This repository is provided **solely for transparency and peer review**.
No reuse is permitted without prior written permission. See `USAGE_NOTICE.txt`.

---

## Contact

**First author:** Irfan Haider — [irfanhaider@mail.dlut.edu.cn](mailto:irfanhaider@mail.dlut.edu.cn)

