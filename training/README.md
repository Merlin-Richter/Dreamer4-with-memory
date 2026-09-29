
Train with saved data (no environment dependency in the trainer/loader):

```sh
python -m training.train_tokenizer --data data/cards.h5 --output runs/tokenizer_cards --batch-size 8 --context-size 16 --workers 2 --device cuda
python -m training.train_tokenizer --data data/ball.h5 --output runs/tokenizer_ball --batch-size 8 --context-size 16 --workers 2 --device cuda
```

`batch-size B` means B clips of T=`context-size` consecutive frames. Both temporal modes process B*T frames: use `--temporal-attention` for `(B,T,...)`; the default reshapes to `(B*T,1,...)`. Image dimensions come from the file.


The dataset indexes every contiguous length-T window; windows can overlap but never cross an episode boundary. Training samples clips uniformly with replacement. Validation uses a fixed random subset without replacement, up to `validation-batches * batch-size` clips; small datasets may give fewer batches or a short last batch. Validation MSE is weighted by pixel count.

`workers > 0` enables PyTorch multiprocess prefetching (two batches per worker); each process opens its own read-only HDF5 handle. CUDA uses pinned host memory and nonblocking transfers. `--workers 0` is a synchronous debugging option.
