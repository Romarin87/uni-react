# Configurations

This directory keeps configs for the retained model families:

- `single_mol`
- `gotennet_s`
- `gotennet_b`
- `gotennet_l`
- `gotennet_s_hat`
- `gotennet_b_hat`
- `gotennet_l_hat`

## Layout

```text
configs/
├── single_mol/
│   ├── geometric.yaml
│   ├── geometric.entry_smoke.json
│   ├── cdft.yaml
│   ├── cdft.entry_smoke.json
│   ├── density.yaml
│   ├── reaction.yaml
│   ├── qm9.yaml
│   └── qm9_all.yaml
├── gotennet_s/
├── gotennet_b/
├── gotennet_l/
├── gotennet_s_hat/
├── gotennet_b_hat/
└── gotennet_l_hat/
```

## Main Configs

- grouped task recipes should be represented as `joint.yaml` task selections
- `reaction.yaml`: reaction triplet task training
- `qm9.yaml`: single-target QM9 task
- `qm9_all.yaml`: multi-target QM9 task
- `gotennet_*/qm9.yaml`: GotenNet S/B/L (and hat) QM9 task with the official-style split/head/optimizer recipe

## Smoke Configs

- `gotennet_l/joint.yaml`: joint task training config

## Typical Usage

```bash
python -m uni_react.train_joint --config configs/gotennet_l/joint.yaml
python -m uni_react.train_reaction --config configs/single_mol/reaction.yaml
python -m uni_react.train_qm9 --config configs/gotennet_l/qm9.yaml
```
