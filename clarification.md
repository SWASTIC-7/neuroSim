# Questions to clarify

## What will be our target network size?
Our scope is to scale upto more than 10 million neurons. But we will scale in steps

## What will be our Synaptic coupling type?
We will follow conductance based synapses as they are more biologicaly accurate. This will be chanllenging though due to requirement of V (membrane potential).

## Uniform vs. heterogeneous neuron parameters?
Currently we will focus on uniform neuron parameters and try to bring that to scale. In future we can add the feature of heterogenous neuron parameters option.

## Timestep resolution needed (0.01ms for full HH accuracy vs. relaxed)?
GeNN" benchmarks used 0.04ms or 0.2ms, whereas "SimHH" used 0.025ms for high biological accuracy. We are currently in discussion to see what we shall adapt.


## Real-time simulation required or offline acceptable?
currently we will work on offline simulation then we will switch to real-time simulation. Aur scope is Real-time simulation.

## Target GPU (if specific architecture in mind)?
Currently we will be working on our laptop's GPU for small area network, then we will switch to other better GPU's like Param Ganga, etc.

## Preferred output format (spike rasters, voltage traces, phase plots)?
Voltage traces. SimHH has implemented a binary output of data , will try to implement that in better and effective way.

## Any specific computational neuroscience tools to interface with?
we want to make a standalone tool so that we can experiment in all directions.

## Vectorized PyTorch implementation... vs Optimized CUDA kernels?
We will be writing optimized cuda kernels.

## Sparse connectivity handling (CSR format...)
CSR is memory efficient but coalescing is the major threat

## Will we be implemeting Modular architecture extensible for future learning mechanisms?
Yes, our long term goal is to add Plasticity rules like STDP (Spike-Timing Dependent Plasticity).


