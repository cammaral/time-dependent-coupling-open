# Quantum magic / SRE-2 patch

Este patch adiciona códigos para calcular **quantum magic** no mesmo estilo dos arquivos `coerence.py`, `entanglment.py`, `enegy.py` e `wigner.py`.

A medida usada é a **Stabilizer Rényi Entropy de ordem 2** para o qubit reduzido do modelo Jaynes-Cummings:

```text
M2 = log(S2/S4)
S2 = 1 + |<X>|^2 + |<Y>|^2 + |<Z>|^2
S4 = 1 + |<X>|^4 + |<Y>|^4 + |<Z>|^4
```

Isso segue a convenção que já aparecia no notebook `quantum_magic_time_dep_jc-2(2).ipynb`.

## Arquivos

- `quantum/non_classicality.py`
  - Mantém `coerence`, `entanglement` e `wigner_negativity`.
  - Adiciona:
    - `stabilizer_renyi_entropy_qubit_from_expectations`
    - `qubit_pauli_expectations_from_states`
    - `quantum_magic`
    - `quantum_magic_details`

- `quantum_magic_cos_var_w.py`
  - Varre `w_list` para `coupling='cos'`.
  - Salva em `results/magic/magic_cos1`, `magic_cos2`, etc.

- `quantum_magic_gauss_varT.py`
  - Varre `T_list` para `coupling='gauss'`.
  - Salva em `results/magic/magic_gauss1`, `magic_gauss2`, etc.

- `plot_quantum_magic_result.py`
  - Carrega uma pasta de resultado e gera figuras em `plots/`.

## Como usar

Copie os arquivos para a raiz do projeto, mantendo a pasta `quantum/`.

Para rodar o caso cosseno:

```bash
python quantum_magic_cos_var_w.py
```

Para rodar o caso gaussiano variando `T`:

```bash
python quantum_magic_gauss_varT.py
```

Depois edite o caminho `save_dir` em `plot_quantum_magic_result.py`, por exemplo:

```python
save_dir = "results/magic/magic_cos1"
```

E rode:

```bash
python plot_quantum_magic_result.py
```

## Saídas principais

Dentro da pasta criada, os arquivos principais são:

- `run_info.txt`: resumo dos parâmetros.
- `args.json`: argumentos iniciais.
- `t.npy`: grade temporal.
- `w_list.npy` ou `T_list.npy`: grade do parâmetro varrido.
- `const.npy`: M2 para Hamiltoniano constante fechado.
- `const_aberto.npy`: M2 para Hamiltoniano constante aberto.
- `var.npy`: matriz M2 para Hamiltoniano variável fechado.
- `var_aberto.npy`: matriz M2 para Hamiltoniano variável aberto.
- `*_details.npz` e `*_details.csv`: `<X>`, `<Y>`, `<Z>`, `S2`, `S4`, pureza e o diagnóstico antigo `1+|X|+|Y|+|Z|`.
- `snapshots_const/` e `snapshots_const_aberto/`: estados reduzidos do qubit em alguns tempos.
