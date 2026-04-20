# Configurations

This directory only keeps configs for the two retained models:

- `single_mol`
- `gotennet_l`

## Layout

```text
configs/
├── single_mol/
│   ├── geometric.yaml
│   ├── geometric.entry_smoke.json
│   ├── cdft.yaml
│   ├── cdft.entry_smoke.json
│   ├── density.yaml
│   └── reaction.yaml
├── gotennet_l/
│   ├── geometric.yaml
│   ├── cdft.yaml
│   ├── density.yaml
│   ├── reaction.yaml
│   └── qm9_gap.yaml
├── finetune_qm9_gap.yaml
└── finetune_qm9_all.yaml
```

## Main Configs

- `geometric.yaml`: geometric structure pretraining
- `cdft.yaml`: electronic-structure pretraining
- `density.yaml`: electron-density pretraining
- `reaction.yaml`: reaction triplet pretraining
- `finetune_qm9_gap.yaml`: single-target QM9 fine-tuning
- `finetune_qm9_all.yaml`: multi-target QM9 fine-tuning
- `gotennet_l/qm9_gap.yaml`: GotenNet-L QM9 gap fine-tuning with the official-style split/head/optimizer recipe

## Smoke Configs

- `single_mol/geometric.entry_smoke.json`: minimal geometric CLI smoke
- `single_mol/cdft.entry_smoke.json`: minimal CDFT CLI smoke

## Typical Usage

```bash
python -m uni_react.train_pretrain_geometric --config configs/single_mol/geometric.yaml
python -m uni_react.train_pretrain_cdft --config configs/gotennet_l/cdft.yaml
python -m uni_react.train_pretrain_density --config configs/gotennet_l/density.yaml
python -m uni_react.train_pretrain_reaction --config configs/single_mol/reaction.yaml
python -m uni_react.train_finetune_qm9 --config configs/gotennet_l/qm9_gap.yaml
```
