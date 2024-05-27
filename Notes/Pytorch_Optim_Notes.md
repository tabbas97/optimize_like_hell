# Pytorch Optimization Notes

1. Torch no_grad must the outer wrapper of profiling. This is a known issue. It has been somewhat fixed/clarified in MosaicML's trainer. But it's still open on Pytorch. [Open Issue](https://github.com/pytorch/pytorch/issues/100241).

2. This whole thing is based on RTX 2070 Super for the most part. An updated version with newer GPU/devices will be made.
