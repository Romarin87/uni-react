# uni-react

`uni-react` is a deep learning codebase for 3D molecular pretraining and property prediction.

This repository is still under active development. The current codebase mainly includes model implementations, training entry points, configs, and tests.

## Overview

```text
uni_react/
├── models/      Supported model families (`single_mol`, `gotennet_l`)
├── tasks/       Task specs, datasets, heads, losses, trainers, entry helpers
├── logger.py    Runtime logging and result writing
├── training/    Shared runtime utilities, schedulers, trainers, accumulators
└── data/        Data pipeline utilities and converters
```

## Status

- Experimental research code
- Interfaces and training workflows may still change
- Documentation is intentionally brief for now
