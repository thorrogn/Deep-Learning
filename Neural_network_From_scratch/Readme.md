# Neural Network Implementation in Python

## Overview
This project implements a flexible multilayer neural network from scratch using NumPy. The implementation includes a fully-connected feedforward neural network with configurable architecture, suitable for binary classification tasks. The network is demonstrated using the XOR problem as an example.

## Features
- Configurable number of hidden layers and neurons
- Xavier/Glorot weight initialization for better convergence
- Mini-batch gradient descent
- Numerical stability improvements
- Progress monitoring with loss and accuracy metrics
- Input validation and error handling

## Requirements
- Python 3.x
- NumPy

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-network-implementation.git
cd neural-network-implementation
```

2. Install the required dependencies:
```bash
pip install numpy
```

## Usage

### Basic Usage
```python
from neural_network import NeuralNetwork

# Create a neural network with 2 inputs, 1 hidden layer with 4 neurons, and 1 output
nn = NeuralNetwork(input_size=2, hidden_layers_sizes=[4], output_size=1)

# Train the network
nn.train(X_train, y_train, epochs=10000, learning_rate=0.1)

# Make predictions
predictions = nn.forward(X_test)
```

### XOR Problem Example
The included example demonstrates the network's ability to learn the XOR function:
```python
# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the network
nn = NeuralNetwork(input_size=2, hidden_layers_sizes=[4], output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1, batch_size=4)
```

## Network Architecture

### Layer Configuration
- Input Layer: Accepts features as input
- Hidden Layers: Configurable number and size
- Output Layer: Single neuron for binary classification

### Key Components
1. **Activation Function**: Sigmoid activation for all layers
2. **Loss Function**: Mean Squared Error (MSE)
3. **Weight Initialization**: Xavier/Glorot initialization
4. **Training Algorithm**: Mini-batch gradient descent

## API Reference

### NeuralNetwork Class

#### Constructor
```python
NeuralNetwork(input_size, hidden_layers_sizes, output_size)
```
- `input_size`: Number of input features
- `hidden_layers_sizes`: List of integers specifying the size of each hidden layer
- `output_size`: Number of output neurons (typically 1 for binary classification)

#### Methods

##### train()
```python
train(X, y, epochs, learning_rate, batch_size=None, verbose=True)
```
- `X`: Input features (numpy array)
- `y`: Target values (numpy array)
- `epochs`: Number of training iterations
- `learning_rate`: Learning rate for gradient descent
- `batch_size`: Size of mini-batches (default: None, using full batch)
- `verbose`: Whether to print training progress (default: True)

##### forward()
```python
forward(X)
```
- `X`: Input features
- Returns: Network predictions

## Error Handling
The implementation includes comprehensive error handling for:
- Invalid input dimensions
- Incorrect data types
- Invalid layer configurations
- Numerical overflow protection

## Performance Considerations
- Uses NumPy for efficient matrix operations
- Implements gradient clipping to prevent exploding gradients
- Includes Xavier initialization for faster convergence
- Supports mini-batch processing for memory efficiency

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Built with NumPy
- Inspired by classic neural network architectures
- Implements best practices for neural network implementation

