# Self-Pruning-Neural-Network-with-Learnable-Gates-CIFAR-10-PyTorch
A PyTorch-based Self-Pruning Neural Network that learns sparsity using differentiable gates and L1 regularization, evaluated on the CIFAR-10 dataset with automated sparsity tracking and visualization.

A deep learning project implementing a **Self-Pruning Neural Network** using learnable gating mechanisms to automatically remove unnecessary connections during training.  
The model is evaluated on the CIFAR-10 dataset and demonstrates controlled sparsity using L1 regularization.


## Project Overview

This project introduces a custom neural network layer called **PrunableLinear**, where each weight has an associated learnable gate.  
The network learns which connections are important and gradually prunes less useful ones during training.

Key concept:

Weights are multiplied by sigmoid-activated gates:

    pruned_weights = weights * sigmoid(gate_scores)

This enables dynamic model compression without manual pruning.

## Key Features

- Custom PrunableLinear layer implemented from scratch
- Learnable pruning mechanism using sigmoid gates
- L1 sparsity regularization
- Automatic sparsity tracking
- CIFAR-10 image classification
- Model checkpoint saving
- Training and testing pipeline
- Gate distribution visualization
- Reproducible experiments

## Technology Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Torchvision
- Scikit-learn

## Dataset

CIFAR-10 Dataset:

- 60,000 images
- 10 classes
- Image size: 32 × 32
- RGB images

Classes:

- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

Dataset automatically downloads during execution.

## Model Architecture

The neural network uses multiple **PrunableLinear** layers.

Each layer contains:

- Weight parameters
- Bias parameters
- Learnable gate scores
- Sigmoid gate activation
- Element-wise pruning

Forward computation:

    gates = sigmoid(gate_scores)
    pruned_weights = weights * gates

## Loss Function

Total loss:

    Total Loss =
        CrossEntropyLoss
        +
        λ × SparsityLoss

Where:

    SparsityLoss =
        Sum of absolute gate values

Lambda values tested:

    0.0001
    0.001
    0.01


## Sparsity Definition

A connection is considered pruned if:

    gate_value < 0.01

Sparsity percentage:

    (# of gates < 0.01 / total gates) × 100


