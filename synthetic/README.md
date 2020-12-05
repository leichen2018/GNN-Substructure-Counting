# Synthetic 

We include five substructure counting tasks: 3-stars, triangles, tailed triangles, chordal cycles and attributed triangles. 3-star is a subgraph-counting task while the remaining are induced-subgraph-counting.

## Data preparation

In the ``./data`` folder, ``*.bin`` stores all graphs and corresponding labels of substructure counting that can be loaded with **DGL 0.4.3**, while ``*.txt`` stores graph indices for training/validation/testing. Dataset1 and Dataset2 stand for datasets of Erdos-Renyi and Random Regular graphs. For details, we refer to ``line 97-127`` in ``dataset_synthetic.py``.

Note that the results of MSE Loss need to be divided by variances as in ``line 264, 267, 268`` of ``main_synthetic.py``. Variances here are different from our reported variances by 9x times for counting tailed triangles and chordal cycles, but it should not influence the final normalized results. Apologies for such inconsistence and we will fix it after the conference.

## LRP preparation

Prepared LRP sequences will be automatically stored in ``./data/lrp_save_path/``. It takes ~5 and ~12 minutes to generate LRP sequences for dataset 1 and 2.

## Scripts for results in Table 1 and 5

- ${DATASET} is from ``1 2``.

- ${TASK} is from ``star triangle tailed_triangle chordal_cycle attributed_triangle``.

- ${LR} is from ``0.001 0.002``.

```
Deep LRP-1-3:

python3.7m main_synthetic.py --dataset ${DATASET} --task ${TASK} --lr ${LR} --seed ${SEED}
```

