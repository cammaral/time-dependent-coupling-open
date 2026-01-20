Saving simulation results
=========================

This project includes helpers to save and reload simulation results in an organized folder structure.

Location
- Results are saved under the `results/` directory by default.
- Each run creates a folder named `<timestamp>_<params>` where `<params>` is a compact encoding of the `params` dict passed to the saver.

What is saved
- `times.npy` — array of times
- `expectations.npy` and optionally `expectations.csv` — expectation values
- `states.qsave` — pickled QuTiP state objects (load with `qload`)
- `<nc_name>.npy` (and optionally `.csv`) — non-classicality measures (if produced)
- `metadata.json` — run metadata and file listing

Quick examples

1) Save a run from `quantum.run.solve`:

```python
from quantum.run import solve

# call your solver as usual, add `save_dir` and `params`
sol = solve(H, psi0, t, c_ops, obs_list, args,
            save_dir='results', params={'model':'myModel','N':2,'g':0.5}, save_nonclassical=True)
```

2) Load saved results and plot:

```python
from utils.io import load_results
from utils.plotting import plot_expectations, plot_non_classicalities

res = load_results('results/20250101T123000_model=myModel_N=2_g=0p5')
plot_expectations(res)
plot_non_classicalities(res)
```

Notes
- Use `qutip.qload` to load `states.qsave` directly if needed.
- Filenames are sanitized; floating point numbers in the folder name use a compact decimal representation.

If you want, I can also add a CLI helper to list result folders and open plots quickly.
