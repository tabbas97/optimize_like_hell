# %%
# Using version 2.3.0 of torch on Python 3.10 as of May 26 2024. 

import torch
print(torch.__version__)

# %% [markdown]
# EXACT VERSION USED : 2.3.0+cu121

# %%
# Example Model - A simple conv neural network with 5 hidden layers

class SampleModel(torch.nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 3, 1)
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
model = SampleModel()

model.cuda() # Move model to GPU

# %% [markdown]
# ## Fusing Model

# %%
# We follow fx graph

from torch.fx import symbolic_trace

symbolic_traced_model = symbolic_trace(model)

print(symbolic_traced_model.graph)

# %%
print(symbolic_traced_model.code)

# %%
# What's happening on eager execution

from torch.profiler import profile, record_function, ProfilerActivity

input = torch.randn(100, 1, 28, 28)
input = input.cuda()

with profile(activities=[
    ProfilerActivity.CUDA, # Will only record the time spent on GPU
    ProfilerActivity.CPU   # Will only record the time spent on CPU. Need both to get the complete picture
    ], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input)
        
prof.export_chrome_trace("trace.json")

# %%
# We will use Torch.compile to compile the model

compiled_model = torch.compile(
    model=model,
    fullgraph=True, # Ideally we want the complete graph on device. Default is set to False
                    # In case the full graph is not compilable, we can set it to False.
    )

with profile(activities=[
    ProfilerActivity.CUDA, # Will only record the time spent on GPU
    ProfilerActivity.CPU   # Will only record the time spent on CPU. Need both to get the complete picture
    ], record_shapes=True) as compiled_prof:
    with record_function("model_inference"):
        compiled_model(input)
        
compiled_prof.export_chrome_trace("compiled_trace.json")

# %%
# Setting up use of tensor cores
torch.set_float32_matmul_precision("high")

# %%
# Compile model again
new_compiled_model = torch.compile(
    model=model,
    fullgraph=True, # Ideally we want the complete graph on device. Default is set to False
                    # In case the full graph is not compilable, we can set it to False.
    )

with profile(activities=[
    ProfilerActivity.CUDA, # Will only record the time spent on GPU
    ProfilerActivity.CPU   # Will only record the time spent on CPU. Need both to get the complete picture
    ], record_shapes=True) as new_compiled_prof:
    with record_function("model_inference"):
        new_compiled_model(input)
        
new_compiled_prof.export_chrome_trace("new_compiled_trace.json")

# %%
# Switching to FP16
torch.set_default_dtype(torch.float16)

# Model to FP16
fp_16_base_model = model.half()
input_fp16 = input.half().cuda()

# Compile model again
fp16_compiled_model = torch.compile(
    model=fp_16_base_model,
    fullgraph=True, # Ideally we want the complete graph on device. Default is set to False
                    # In case the full graph is not compilable, we can set it to False.
    )

with profile(activities=[
    ProfilerActivity.CUDA, # Will only record the time spent on GPU
    ProfilerActivity.CPU   # Will only record the time spent on CPU. Need both to get the complete picture
    ], record_shapes=True) as fp16_compiled_prof:
    with record_function("model_inference"):
        fp16_compiled_model(input_fp16)
        
fp16_compiled_prof.export_chrome_trace("fp16_compiled_trace.json")

# %% [markdown]
# FP16 cuts down processing time in half
# 
# TODO : Need to investigate the effect of FP32 in RTX 20 series Tensor cores. The 2nd gen Tensor Cores. 
# Preliminary assessment : Leaving the dtype on FP32 effectively does not use the Tensor Cores. We would need to drop into FP16, INT8, INT4, INT1.
# 
# Note: Not all Tensor cores are the same. The 40 series has 4th gen Tensor Cores. Blackwell is on 5th gen. Refer to this sheet for the capabilities of each of the generation of Tensor Cores
# 
# 

# %% [markdown]
# Refer to this blog for more information on the GPU family vs types supported
# 
# https://bruce-lee-ly.medium.com/nvidia-tensor-core-preliminary-exploration-10618787615a
# 
# ![alt text](image.png)

# %%
# Leaving in a cell to be run on the latest gen on L4 GPU - Hopper Architecture

# Hopper natively supports TF32 on Tensor Cores. That should give the next big boost in performance.
# We will see this differnce in profiling on previous profiled traces.



# %% [markdown]
# ### Result of Compiling
# 
# The major reason to compile is:
# 1. To avoid the need to sync any of the operations back to the CPU
# 2. Automatically introduces what you could consider true async operation at the device(GPU) level. The graph would have a separate location for the CUDA memsync to happen back to the CPU. 
# 

# %%
# Print the model_inference time for all the traces

print("Eager Execution Time: ", prof.key_averages().table(row_limit=1))
print("Compiled Execution Time: ", compiled_prof.key_averages().table(row_limit=1))
print("New Compiled Execution Time: ", new_compiled_prof.key_averages().table(row_limit=1))
print("FP16 Compiled Execution Time: ", fp16_compiled_prof.key_averages().table(row_limit=1))

# %% [markdown]
# ### Overall Times - RTX 2070 Super
# 
# Note: The GPU is also running 3 displays and might have effect on these numbers
# 
# | Mode | Self CPU Total Time | Self CUDA Total Time |
# |------|---------------------|----------------------|
# |Eager Execution|128.710ms|60.790ms|
# |Basic Compile  |6.446ms|10.682ms|
# |Compiled Tensor Core Enable (RTX2070 Super)| 7.200ms | 10.224ms |
# |Compiled Tensor Core FP16 (RTX2070 Super)| 3.584ms | 2.612ms |
# 
# Eager Execution Time:
# 
# |               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls |
# |--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
# |    eager_execution |        0.65% |    842.000us |      100.00% |    128.704ms |    128.704ms |      0.000us |        0.00%  |     3.364ms |      3.364ms  |           1  |
# 
# Compiled Execution Time:
# 
# |               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls |
# |--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
# |    model_inference |        2.79% |    180.000us |       48.74% |      3.142ms |      3.142ms |      0.000us |        0.00%  |     4.293ms |      4.293ms  |           1  |
# 
# New Compiled Execution Time:
# 
# |               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls |
# |--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
# |    model_inference |        4.65% |    335.000us |       53.57% |      3.857ms |      3.857ms |      0.000us |        0.00%  |     4.283ms |      4.283ms  |           1  |
# 
# FP16 Compiled Execution Time:
# 
# |               Name |   Self CPU % |     Self CPU |  CPU total % |    CPU total | CPU time avg |    Self CUDA |  Self CUDA %  |  CUDA total | CUDA time avg |   # of Calls  |
# |--------------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|---------------|--------------|
# |    model_inference |        5.47% |    196.000us |       92.97% |      3.332ms |      3.332ms |      0.000us |        0.00%  |     1.180ms |      1.180ms  |           1   |
# 

# %% [markdown]
# ## CUDA Graph Mode Execution
# 
# CUDA graph mode execution offers a more streamlined graph operation onboard the GPU. This is turned off by default in Pytorch for compatibility and memory reasons.
# CUDA graph mode is known to consume slightly higher RAM than the normal compiled model.

# %%
# CUDA Graph mode execution with FP16

fp16_cuda_graph = torch.compile(
    model=fp_16_base_model,
    fullgraph=True, # Ideally we want the complete graph on device. Default is set to False
                    # In case the full graph is not compilable, we can set it to False.
    mode="reduce-overhead"
    )


# %%
with torch.no_grad():
    with profile(activities=[
        ProfilerActivity.CUDA, # Will only record the time spent on GPU
        ProfilerActivity.CPU   # Will only record the time spent on CPU. Need both to get the complete picture
        ], record_shapes=True) as fp16_cuda_graph_prof:
        with record_function("model_inference"):
            fp16_cuda_graph(input_fp16)



# %%

fp16_cuda_graph_prof.export_chrome_trace("fp16_cuda_graph_trace.json")

# %%
print("CUDA Graph Mode Execution Time: ", fp16_cuda_graph_prof.key_averages().table(row_limit=1))

# %% [markdown]
# This does not necessarily affect the overall time but does reduce the device(GPU) side time. The effects of this would be observed better with a pipeline of calls being made to the model

# %%
# Torch.compile vs torch.cuda.make_graphed_callables

# We will use torch.cuda.make_graphed_callables to create a graphed callable and compare it with torch.compile

graphed_callable = torch.cuda.make_graphed_callables(model, (input_fp16,))

# %% [markdown]
# ## Quantization
# 
# Quantization is the next big speedup that we can observe. The effects of quantization are generally seen at higher model sizes.
# 
# 

# %%
# 


