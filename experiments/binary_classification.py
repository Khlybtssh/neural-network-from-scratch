import sys
import os

# Add project root to PYTHONPATH so absolute imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_toy_data
from utils.visualization import plot_history
from engine.model import NeuralNetwork
from optimizers.adam import Adam
from layers.dense import Dense
from layers.activation import Activation
from layers.dropout import Dropout

if __name__ == "__main__":
    
    print("Loading Toy Dataset...")
    X_train, y_train, X_val, y_val = get_toy_data()
    
    model = NeuralNetwork(loss_name='bce', optimizer=Adam(learning_rate=0.01))
    
    model.add(Dense(input_size=2, output_size=16, initialization='he', l2_lambda=0.01))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2))
    
    model.add(Dense(input_size=16, output_size=8, initialization='he', l2_lambda=0.01))
    model.add(Activation('relu'))
    
    model.add(Dense(input_size=8, output_size=1, initialization='xavier'))
    model.add(Activation('sigmoid'))
    
    print("Training Model...")
    history = model.train(X_train, y_train, 
                          epochs=150, 
                          batch_size=32,
                          X_val=X_val, y_val=y_val, 
                          early_stopping_patience=20)
                          
    print("Plotting Metrics...")
    plot_history(history)
