# GNN-Substructure-Counting

The official implementation of ''Can Graph Neural Networks Count Substructures?'' at NeurIPS 2020.

Authors: [Zhengdao Chen](https://cims.nyu.edu/~chenzh/) (NYU), [Lei Chen](https://leichen2018.github.io) (NYU), [Soledad Villar](https://cims.nyu.edu/~villar/) (JHU), [Joan Bruna](https://cims.nyu.edu/~bruna/) (NYU).

## Dependencies

Core packages:
```
pytorch 1.5.0
dgl 0.4.3
torch-sparse 0.6.5
torch-scatter 2.0.4
ogb 1.1.1
scipy 1.2.1
```
Note that

1) **DGL** released a massive update of APIs in 0.5, due to which the implementation is in need of modification accordingly if you would like to install a more recent version. We refer to an official API guidebook for 0.4.x [1] and an official blog revealing ``what's new in 0.5 release`` [2].

2) Following the official instructions [3] to install **torch-sparse** according to versions of your pytorch and cuda. Note that **torch-scatter** is necessary althougth we do not explicitly import it in the implementation.

3) Other packages with higher verisions should be compatible. If you are with any questions, please do not hesitate to email us via ``lc3909@nyu.edu``.

```
[1] https://docs.dgl.ai/en/0.4.x/api/python/graph.html
[2] https://www.dgl.ai/release/2020/08/26/release.html
[3] https://github.com/rusty1s/pytorch_sparse
```

## For citation
```
@article{chen2020can,
  title={Can graph neural networks count substructures?},
  author={Chen, Zhengdao and Chen, Lei and Villar, Soledad and Bruna, Joan},
  journal={arXiv preprint arXiv:2002.04025},
  year={2020}
}
```
