# Laya Smoke Test

Safe default (small metadata requests only, no model download or training):

```bash
bash experiments/laya_smoke_test/run.sh
```

If an existing environment contains PyTorch, the command also runs a tiny CPU loss/gradient check. `config.json` is the recommended starting profile for implementing a one-step full-model VRAM test; it is not consumed by the upstream notebook and is not evidence that training fits.

To prepare a clean environment later, choose the correct PyTorch CUDA wheel from the [official installer](https://pytorch.org/get-started/locally/), then install the pinned source checkout and data dependencies. Do not run the upstream notebook unchanged on one GPU: it asserts two GPUs and invokes `torchrun --nproc_per_node=2`.
