# Questions to clarify

## What will be our target network size?
Our scope is to scale up to more than 10 million neurons. But we will scale in steps. This is a long-term upper bound, not an initial target. 

## What will be our Synaptic coupling type?
We will follow conductance-based synapses as they are more biologically accurate. This will be challenging, though, due to the requirement of V (membrane potential).

## Uniform vs. heterogeneous neuron parameters?
Currently, we will focus on uniform neuron parameters and aim to scale them accordingly. In future, we can add the feature of the heterogeneous neuron parameters option.

## Timestep resolution needed (0.01ms for full HH accuracy vs. relaxed)?
GeNN" benchmarks used 0.04ms or 0.2ms, whereas "SimHH" used 0.025ms for high biological accuracy. We are currently in discussion to see what we shall adapt.

## Real-time simulation required or offline acceptable?
Currently, we will work on offline simulation. If the performance and hardware limits allow us to approach real-time, we would be happy to experiment in that direction.

## Target GPU (if specific architecture in mind)?
We are currently targeting NVIDIA GPUs, as PARAM Ganga supports NVIDIA's architecture, and we are writing CUDA kernels for this purpose.

## Preferred output format (spike rasters, voltage traces, phase plots)?
Voltage traces. SimHH has implemented a binary output of data and will strive to implement it in a more effective way. Phase plots are also an interesting output format, helping us understand the structure and dynamics of evolution in a system. We would also like to work on that.

## Are there any specific computational neuroscience tools that can be used to interface with?
We aim to create a standalone tool that allows us to experiment in all directions as well. However, extensibility to other tools is always a valuable feature to have. We would work on that, but it is not our priority at this time. We currently have academic goals for this project, but we may work on Python bindings or interfacing with NEURON or Brian2 in future.

## Vectorised PyTorch implementation... vs Optimised CUDA kernels?
We will be writing optimised CUDA kernels.

## Sparse connectivity handling (CSR format...)
CSR is memory efficient, but coalescing is the major threat

## Will we implement a Modular architecture that is extensible for future learning mechanisms?
Yes, our long-term goal is to incorporate Plasticity rules, such as STDP (Spike-Timing Dependent Plasticity). 


