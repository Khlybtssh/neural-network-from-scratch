import numpy as np

def get_toy_data():
    """Generates synthetic circles dataset for binary classification"""
    np.random.seed(42)
    n_samples = 400
    noise = 0.1
    
    t1 = np.linspace(0, 2 * np.pi, n_samples // 2)
    x1 = np.vstack((np.cos(t1), np.sin(t1))).T + np.random.randn(n_samples // 2, 2) * noise
    y1 = np.zeros((n_samples // 2, 1))
    
    t2 = np.linspace(0, 2 * np.pi, n_samples // 2)
    x2 = 0.5 * np.vstack((np.cos(t2), np.sin(t2))).T + np.random.randn(n_samples // 2, 2) * noise
    y2 = np.ones((n_samples // 2, 1))
    
    X = np.vstack((x1, x2))
    y = np.vstack((y1, y2))
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    split = int(0.8 * n_samples)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    return X_train, y_train, X_val, y_val
