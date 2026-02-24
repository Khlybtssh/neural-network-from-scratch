Neural Network from Scratch (NumPy)

A modular, lightweight Deep Learning framework built entirely from scratch using Python and NumPy. This project is designed for educational purposes to demonstrate the inner workings of artificial neural networks, including forward propagation, backpropagation, optimization, and regularization, without relying on heavy frameworks like TensorFlow or PyTorch.



✨ Features

Modular Architecture: Components are separated into distinct modules (layers, optimizers, losses, etc.) mimicking standard deep learning libraries.



Layers: \* Dense (Fully Connected) with configurable weight initializations (He, Xavier, Random Normal) and L2 Regularization.



Activation supporting ReLU, Sigmoid, Tanh, and Softmax.



Dropout for robust training and overfitting prevention.



Optimizers:



Vanilla SGD with optional Momentum.



Adam (Adaptive Moment Estimation).



Loss Functions:



Mean Squared Error (MSE).



Binary Cross-Entropy (BCE).



Categorical Cross-Entropy (CCE).



Training Engine:



Custom training loop with mini-batching.



Validation support and Early Stopping.



Metrics tracking (Loss and Accuracy).



Utilities: Synthetic data generation for testing (e.g., noisy circles) and Matplotlib-based history plotting.



📂 Project Structure

Plaintext

neural-network-from-scratch/

│

├── core/

│   └── layer.py                 # Base Layer abstract class

│

├── layers/

│   ├── dense.py                 # Fully connected layer

│   ├── activation.py            # Activation functions (ReLU, Sigmoid, etc.)

│   └── dropout.py               # Dropout layer implementation

│

├── losses/

│   └── losses.py                # Loss functions and their derivatives

│

├── optimizers/

│   ├── sgd.py                   # Stochastic Gradient Descent

│   └── adam.py                  # Adam optimizer

│

├── engine/

│   └── model.py                 # NeuralNetwork class \& training loop

│

├── utils/

│   ├── metrics.py               # Accuracy calculations

│   ├── data.py                  # Toy dataset generation

│   └── visualization.py         # Matplotlib plotting functions

│

├── experiments/

│   └── binary\_classification.py # Example script tying everything together

│

├── README.md

└── requirements.txt

🚀 Getting Started

Prerequisites

All you need is Python 3.x and a couple of standard scientific libraries.



Bash

pip install -r requirements.txt

(Note: requirements.txt only requires numpy and matplotlib)



Basic Usage

Here is an example of how to construct, train, and evaluate a simple Multi-Layer Perceptron (MLP) for binary classification using the modular components:



Python

from engine.model import NeuralNetwork

from layers.dense import Dense

from layers.activation import Activation

from layers.dropout import Dropout

from optimizers.adam import Adam

from utils.data import get\_toy\_data

from utils.visualization import plot\_history



\# 1. Load Data

X\_train, y\_train, X\_val, y\_val = get\_toy\_data()



\# 2. Initialize Model \& Optimizer

optimizer = Adam(learning\_rate=0.01)

model = NeuralNetwork(loss\_name='bce', optimizer=optimizer)



\# 3. Build Architecture

model.add(Dense(input\_size=2, output\_size=16, initialization='he', l2\_lambda=0.01))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Dense(input\_size=16, output\_size=8, initialization='he', l2\_lambda=0.01))

model.add(Activation('relu'))



model.add(Dense(input\_size=8, output\_size=1, initialization='xavier'))

model.add(Activation('sigmoid'))



\# 4. Train Model

history = model.train(

&nbsp;   X\_train, y\_train, 

&nbsp;   epochs=150, 

&nbsp;   batch\_size=32,

&nbsp;   X\_val=X\_val, 

&nbsp;   y\_val=y\_val, 

&nbsp;   early\_stopping\_patience=20

)



\# 5. Visualize Results

plot\_history(history)

🛠️ How It Works (Under the Hood)

Forward Pass: Data flows through the network. Each layer stores its input (and sometimes specific variables like dropout masks) to use later during backpropagation.



Loss Calculation: The total error is calculated by comparing predictions to the true labels, adding any L2 regularization penalties.



Backward Pass (Backpropagation): The derivative of the loss is computed and passed backward through the network. The Chain Rule is applied at each step to calculate weight gradients (dW) and bias gradients (db).



Optimization: The chosen optimizer updates the network's weights based on the computed gradients to minimize the loss.



🤝 Contributing

Contributions are welcome! If you'd like to add new features like Convolutional Layers, Batch Normalization, or new optimization algorithms (like RMSprop), feel free to fork the repository and submit a pull request.

