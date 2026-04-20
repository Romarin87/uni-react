# GotenNet-L Notes

This directory vendors the public GotenNet implementation and keeps a short
record of how the paper's `S / B / L` variants map onto the exposed code knobs.

## Paper-to-Code Mapping

From the GotenNet paper appendix, the QM9 model depth variants are:

- `S`: 4 interaction blocks
- `B`: 6 interaction blocks
- `L`: 12 interaction blocks

In this codebase, that corresponds to:

- `num_layers` in [backbone.py](/Users/bwli/Desktop/test/uni-react/uni_react/models/gotennet_l/backbone.py)
- `se3_layer` in the task YAMLs under [configs/gotennet_l](/Users/bwli/Desktop/test/uni-react/configs/gotennet_l)

## Hat vs Non-Hat

The paper distinguishes plain `GotenNet` from the hat variants `GotenNet^`,
described as using shared coefficients for the spherical harmonic outputs.

In the public implementation, the closest code-level switch is:

- non-hat:
  - `sep_dir=True`
  - `sep_tensor=True`
- hat (`^`):
  - `sep_dir=False`
  - `sep_tensor=False`

This interpretation was validated by matching parameter counts against the
appendix table:

- `S^`:
  - 4 layers
  - `sep_dir=False`
  - `sep_tensor=False`
  - about `6.08M`
  - paper reports `6.1M`
- `B^`:
  - 6 layers
  - `sep_dir=False`
  - `sep_tensor=False`
  - about `9.11M`
  - paper reports `9.2M`
- `L^`:
  - 12 layers
  - `sep_dir=False`
  - `sep_tensor=False`
  - about `18.18M`
  - paper reports `18.3M`

For comparison, the corresponding non-hat parameter counts from the same code
path are much larger:

- `S`: about `7.66M`
- `B`: about `11.48M`
- `L`: about `22.92M`

## Important Implication

If the goal is to reproduce the paper's `GotenNetL^` results, then setting only
`num_layers=12` is not enough. The hat/shared-coefficient variant also needs:

- `sep_dir=False`
- `sep_tensor=False`

If the goal is instead to match the currently published Hugging Face `QM9/base`
checkpoint, that checkpoint is:

- 6 layers
- closer to the non-hat/base setting
- not the paper's `L^` variant

The public Hugging Face QM9 checkpoints also make one more thing clear:

- `QM9 small`
  - `n_interactions=4`
  - `sep_dir=True`
  - `sep_tensor=True`
  - `layernorm` not set in checkpoint metadata
  - `steerable_norm` not set in checkpoint metadata
- `QM9 base`
  - `n_interactions=6`
  - `sep_dir=True`
  - `sep_tensor=True`
  - `layernorm` not set in checkpoint metadata
  - `steerable_norm` not set in checkpoint metadata

In practice that means the released QM9 `small/base` checkpoints follow the
public non-hat setting for `sep_dir/sep_tensor`, while leaving
`layernorm/steerable_norm` at the implementation defaults rather than explicitly
turning them on.

## Current Repo Status

At the moment, this repo has already been updated so that `gotennet_l` uses:

- `12` interaction blocks
- `layernorm=""`
- `steerable_norm=""`

So the current backbone is aligned with a non-hat 12-layer `GotenNetL`-style
setting rather than the paper's shared-coefficient `GotenNetL^`.

As long as the backbone still keeps:

- `sep_dir=True`
- `sep_tensor=True`

it remains closer to non-hat `GotenNetL` than to `GotenNetL^`.
