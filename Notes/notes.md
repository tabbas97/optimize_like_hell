# Notes

Things to cover :

1. What is the objective you want to target ?
    1. Throughput
        - Look elsewhere in the pipeline first
        - Check pre and post processing
        - Model modifications - Out of Scope - Separate Blog - Memory Bound vs Compute Bound
    2. Latency
2. What is the kind of model ?
    1. Purely Convolutional Model
    2. LLM
        1. Bidirectional Model
        2. Word by Word Regressive Model (Throughput / First Word Latency)

    Will only focus on the latest versions of each of the frameworks

3. Model Framework - Does it have to be kept in the source framework ?
    1. Pytorch - Available optimizations
    2. Tensorflow
    3. Jax

4. Conversion to serving framework
    1. TensorRT
    2. OpenVINO

Writing a GPU Kernel / Optimizing it

Serving modalities (Will likely be a network protocol - ease of use/industry standard)
