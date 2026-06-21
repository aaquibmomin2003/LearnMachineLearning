import numpy as np

# =====================================================================
# 1. SETUP DATA (Same as before)
# =====================================================================
X = np.array([
    [0.1, 0.9],
    [0.8, 0.2],
    [0.7, 0.8],
    [0.2, 0.3]
])

Y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# =====================================================================
# 2. INITIALIZE PARAMETERS (2 Layers)
# =====================================================================
np.random.seed(42)

input_dim = 2    # D = 2
hidden_dim = 4   # H = 4 (4 neurons in our hidden layer)
num_classes = 3  # C = 3

# Layer 1 Parameters (He Initialization)
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim))

# Layer 2 Parameters (Xavier Initialization)
W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(1.0 / hidden_dim)
b2 = np.zeros((1, num_classes))

learning_rate = 0.2
epochs = 300

print("--- TRAINING 2-LAYER NETWORK ---")

# =====================================================================
# 3. THE TRAINING LOOP
# =====================================================================
for epoch in range(epochs):
    num_samples = X.shape[0]
    
    # --- STEP A: FORWARD PASS ---
    # Layer 1 (Hidden)
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU Activation
    
    # Layer 2 (Output)
    Z2 = np.dot(A1, W2) + b2
    # Softmax Activation
    shift_Z2 = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_Z2 = np.exp(shift_Z2)
    Y_pred = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
    
    # --- STEP B: COMPUTE CROSS-ENTROPY LOSS ---
    loss = -np.sum(Y_true * np.log(np.clip(Y_pred, 1e-15, 1.0))) / num_samples
    
    # --- STEP C: BACKPROPAGATION (The Chain Rule) ---
    # 1. Gradient at Output Layer (Same clean subtraction!)
    dZ2 = (Y_pred - Y_true) / num_samples
    
    # 2. Gradients for Layer 2 Parameters
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # 3. Backpropagate error to Hidden Layer Output
    dA1 = np.dot(dZ2, W2.T)
    
    # 4. Backpropagate through the ReLU function
    # If the original input Z1 was <= 0, the gradient stops (becomes 0)
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0
    
    # 5. Gradients for Layer 1 Parameters
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # --- STEP D: UPDATE ALL PARAMETERS ---
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1:03d} | Total Loss: {loss:.5f}")

print("\n--- TRAINING COMPLETE ---")
# Test on a sample
test_sample = np.array([[0.75, 0.21]])
t_Z1 = np.dot(test_sample, W1) + b1
t_A1 = np.maximum(0, t_Z1)
t_Z2 = np.dot(t_A1, W2) + b2
t_exp = np.exp(t_Z2 - np.max(t_Z2))
probs = t_exp / np.sum(t_exp)
print(f"Prediction for [0.75, 0.21]: {probs[0]}")
print(f"Predicted Class: {np.argmax(probs)}")