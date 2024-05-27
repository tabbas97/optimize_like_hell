# Overall Times

| Mode | Self CPU Total Time | Self CUDA Total Time |
|------|---------------------|----------------------|
|Eager Execution|128.710ms|60.790ms|
|Basic Compile  |6.446ms|10.682ms|
|Compiled Tensor Core Enable (RTX2070 Super)| 7.200ms | 10.224ms |
|Compiled Tensor Core FP16 (RTX2070 Super)| 3.584ms | 2.612ms |

Eager Execution Time:

|               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls |
|--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
|    eager_execution |        0.65% |    842.000us |      100.00% |    128.704ms |    128.704ms |      0.000us |        0.00%  |     3.364ms |      3.364ms  |           1  |

Compiled Execution Time:

|               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls |
|--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
|    model_inference |        2.79% |    180.000us |       48.74% |      3.142ms |      3.142ms |      0.000us |        0.00%  |     4.293ms |      4.293ms  |           1  |

New Compiled Execution Time:

|               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls |
|--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
|    model_inference |        4.65% |    335.000us |       53.57% |      3.857ms |      3.857ms |      0.000us |        0.00%  |     4.283ms |      4.283ms  |           1  |

FP16 Compiled Execution Time:

|               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls  |
|--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
|    model_inference |        5.47% |    196.000us |       92.97% |      3.332ms |      3.332ms |      0.000us |        0.00%  |     1.180ms |      1.180ms  |           1   |
