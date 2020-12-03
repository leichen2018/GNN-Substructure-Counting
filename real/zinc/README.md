These are scripts for ZINC results in Table 9.

Deep LRP-7-1 with performance of 0.317±0.031:

python3.7m main_zinc.py --num_layer 20 --hid_dim 8 --mlp --bn --batch_size 64 --num_workers 8 --lr 1e-3 --epochs 1000 --scheduler --lrp_length 8 --lrp_depth 7 --lrp_width 1 --alldegree --seed ${SEED}


Deep LRP-7-1 with performance of 0.244±0.012:

python3.7m main_zinc.py --num_layer 24 --hid_dim 32 --mlp --bn --batch_size 64 --num_workers 8 --lr 1e-3 --epochs 1000 --scheduler --lrp_length 8 --lrp_depth 7 --lrp_width 1 --alldegree --seed ${SEED}


Deep LRP-5-1 with performance of 0.256±0.033:

python3.7m main_zinc.py --num_layer 10 --hid_dim 128 --mlp --bn --batch_size 64 --num_workers 8 --lr 1e-3 --epochs 1000 --scheduler --lrp_length 6 --lrp_depth 5 --lrp_width 1 --alldegree --seed ${SEED}


Deep LRP-7-1 with performance of 0.223±0.008:

python3.7m main_zinc.py --num_layer 10 --hid_dim 128 --mlp --bn --batch_size 64 --num_workers 8 --lr 1e-3 --epochs 1000 --scheduler --lrp_length 8 --lrp_depth 7 --lrp_width 1 --alldegree --seed ${SEED}
