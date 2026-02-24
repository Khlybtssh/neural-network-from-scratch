import numpy as np
from losses.losses import LossFunction
from utils.metrics import compute_accuracy

class NeuralNetwork:
    def __init__(self, loss_name, optimizer):
        self.layers = []
        self.loss_func = LossFunction(loss_name)
        self.optimizer = optimizer
        self.is_categorical = (loss_name == 'cce')
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
        
    def backward(self, loss_derivative):
        error = loss_derivative
        for layer in reversed(self.layers):
            error = layer.backward(error)
            
    def optimize(self):
        if hasattr(self.optimizer, 'increment_t'):
            self.optimizer.increment_t()
            
        for layer in self.layers:
            self.optimizer.update(layer)
            
    def compute_l2_loss(self):
        l2_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'l2_lambda') and layer.l2_lambda > 0:
                l2_loss += (layer.l2_lambda / 2) * np.sum(layer.weights ** 2)
        return l2_loss
        
    def train(self, X_train, y_train, epochs, batch_size,
              X_val=None, y_val=None, early_stopping_patience=None):
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        best_val_loss = float('inf')
        patience_counter = 0
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 1. Forward Pass
                y_pred_batch = self.forward(X_batch, training=True)
                
                # 2. Compute Loss
                batch_loss = self.loss_func.compute(y_batch, y_pred_batch) + self.compute_l2_loss()
                epoch_loss += batch_loss * X_batch.shape[0]
                
                # 3. Backward Pass (Compute Gradients)
                loss_prime = self.loss_func.derivative(y_batch, y_pred_batch)
                self.backward(loss_prime)
                
                # 4. Update Weights
                self.optimize()
                
            epoch_loss /= num_samples
            train_losses.append(epoch_loss)
            
            y_train_pred = self.forward(X_train, training=False)
            train_acc = compute_accuracy(y_train, y_train_pred, self.is_categorical)
            train_accs.append(train_acc)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = self.loss_func.compute(y_val, y_val_pred) + self.compute_l2_loss()
                val_acc = compute_accuracy(y_val, y_val_pred, self.is_categorical)
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {epoch_loss:.4f} - Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                      
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered at epoch {epoch+1}")
                            break
            else:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} - Acc: {train_acc:.4f}")
                
        return {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs
        }
