# Custom Neural Network Implementation

## Overview
This repository contains a modular implementation of a feedforward neural network from scratch using NumPy. The implementation includes support for multiple hidden layers, various activation functions, and mini-batch gradient descent with backpropagation.

## Features
- Configurable network architecture with any number of hidden layers
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Mini-batch gradient descent
- Training history tracking and visualization
- Model saving and loading capabilities
- Interactive user input for network configuration

## Technical Concepts

### Neural Network Architecture
- **Input Layer**: First layer that receives the raw input features
- **Hidden Layers**: Intermediate layers where feature transformation occurs
- **Output Layer**: Final layer that produces the network's predictions
- **Neurons**: Basic units that perform weighted sum of inputs and apply activation function

### Activation Functions
1. **ReLU (Rectified Linear Unit)**
   - Formula: f(x) = max(0, x)
   - Derivative: f'(x) = 1 if x > 0, else 0
   - Best for hidden layers
   - Helps prevent vanishing gradient problem

2. **Sigmoid**
   - Formula: f(x) = 1 / (1 + e^(-x))
   - Derivative: f'(x) = f(x) * (1 - f(x))
   - Output range: [0, 1]
   - Useful for binary classification output layer

3. **Tanh**
   - Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Derivative: f'(x) = 1 - tanh^2(x)
   - Output range: [-1, 1]
   - Zero-centered output

### Training Process
1. **Forward Propagation**
   - Computes layer outputs sequentially
   - Each layer performs: output = activation(weights * input + bias)

2. **Backpropagation**
   - Computes gradients using chain rule
   - Updates weights and biases to minimize loss
   - Formula: weight_update = learning_rate * input.T * delta

3. **Mini-batch Gradient Descent**
   - Processes data in small batches
   - Balances computational efficiency and update frequency
   - Helps avoid local minima

### Loss Function
- Uses Mean Squared Error (MSE)
- Formula: MSE = (1/n) * Σ(y_pred - y_true)²
- Measures prediction accuracy

## Usage

### Basic Implementation
```python
# Create network
layer_sizes = [4, 6, 4, 3]  # Input -> Hidden1 -> Hidden2 -> Output
activations = ['relu', 'relu', 'sigmoid']
nn = NeuralNetwork(layer_sizes, activations)

# Train
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=32)

# Predict
predictions = nn.predict(X_test)
```

### Interactive Usage
Run the main script and follow the prompts to:
1. Define network architecture
2. Set training parameters
3. Train the model
4. Visualize results
5. Save the model

## Key Parameters

### Network Configuration
- **layer_sizes**: List of integers defining neurons in each layer
- **activations**: List of activation functions for each layer

### Training Parameters
- **epochs**: Number of complete passes through the training data
- **learning_rate**: Step size for gradient descent updates (typically 0.01-0.001)
- **batch_size**: Number of samples per gradient update

## Implementation Details

### Layer Class
- Manages individual layer operations
- Stores weights, biases, and activation functions
- Handles forward and backward passes for the layer

### NeuralNetwork Class
- Coordinates multiple layers
- Manages training process
- Provides utility functions (save/load/plot)

## Error Handling
- Validates input dimensions
- Checks activation function compatibility
- Ensures proper layer connectivity

## Visualization
- Plots training loss over time
- Shows accuracy progression
- Helps in hyperparameter tuning

## File Structure
```
neural_network/
│
├── classes/
│   ├── layer.py          # Layer class implementation
│   └── neural_network.py # Main NeuralNetwork class
│
└── main.py              # Interactive implementation
```

## Future Improvements
1. Additional activation functions (LeakyReLU, ELU)
2. More loss functions (Cross-entropy, Huber)
3. Regularization techniques (L1/L2, Dropout)
4. Learning rate scheduling
5. Early stopping

## Requirements
- NumPy
- Matplotlib
- Python 3.x

## License
MIT License - Feel free to use and modify as needed.
