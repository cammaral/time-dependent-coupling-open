# Consolidated Open-TDJC paper dataset

This directory is a non-destructive physical copy of the datasets mapped to
`paper_draft-2.tex` plus the one-channel dissipation studies.

## Main structure

- `article_both_losses/`: article cases with kappa=0.1 and gamma_phi=0.01.
- `only_dephasing/`: kappa=0 and gamma_phi=0.01.
- `only_cavity_damping/`: kappa=0.1 and gamma_phi=0.
- `specific_both_losses/`: exact parameter runs with both losses active.
- `plot_article_and_specific_results.ipynb`: article results, requested
  specific points, and two-column DAT generation/plotting.
- `plot_separate_decay_channels.ipynb`: isolated-loss comparisons and the six
  standalone coherence density maps.
- `notebooks/`: Python scripts used internally by the two root notebooks.
- `requested_outputs/`: generated DAT files and the additional requested PNGs.
- `provenance/`: paper source and simulation/post-processing scripts.

Within each regime, data are separated by resource and coupling profile:
`wigner`, `coherence`, `magic_W_half`, `entanglement`, followed by
`gaussian_zeta`, `gaussian_T`, or `cosine_omega`.

## Magic

The scientific magic result is `W_half`.  The original `M2` arrays are copied
inside `raw_dynamics_legacy_M2` solely to preserve the original simulations.
They are not used by either notebook.

## Expected values

Where available, the package includes g(t), <X>, <Y>, <Z>, <N>, <N^2>, and
Var(N), in both NPY/NPZ and CSV formats.

## Copy summary

- Manifest entries: 123
- Files copied: 44859
- Bytes copied: 6192754119

See `MANIFEST.csv` for the original path of every copied dataset group.
