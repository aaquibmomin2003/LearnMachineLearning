import numpy as np

# =====================================================================
# 1. SETUP THE DATASET
# =====================================================================
# Let's say we have 4 items. Each item has 2 physical features (e.g., size, weight).
# We want to classify them into 3 distinct categories.
X = np.array([
    [0.1, 0.9],
    [0.8, 0.2],
    [0.7, 0.8],
    [0.2, 0.3]
])

# One-Hot Encoded Targets (The Ground Truth 'Y')
# Item 1 is Class 0, Item 2 is Class 1, Item 3 is Class 2, Item 4 is Class 0
Y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# =====================================================================
# 2. INITIALIZE NETWORK PARAMETERS
# =====================================================================
np.random.seed(42)  # Keeps results identical every time you run it

input_dim = 2   # 2 input features
num_classes = 3  # 3 output classes

# Weights matrix (Shape: 2 x 3) and Bias vector (Shape: 1 x 3)
W = np.random.randn(input_dim, num_classes) * 0.01
b = np.zeros((1, num_classes))

# Training configuration
learning_rate = 0.5
epochs = 200

print("--- STARTING TRAINING ENGINE ---")
print(f"Initial Weights:\n{W}\n")

# =====================================================================
# 3. THE TRAINING LOOP
# =====================================================================
for epoch in range(epochs):
    num_samples = X.shape[0]
    
    # --- STEP A: FORWARD PASS (The Logits) ---
    # Z = X * W + b
    Z = np.dot(X, W) + b
    
    # --- STEP B: THE SOFTMAX ACTIVATION ---
    # Shift invariance trick: subtract max(Z) to avoid numerical overflow
    shift_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shift_Z)
    Y_pred = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    # --- STEP C: COMPUTE CROSS-ENTROPY LOSS ---
    # Clip values to avoid log(0) calculation errors
    Y_pred_clipped = np.clip(Y_pred, 1e-15, 1.0 - 1e-15)
    loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / num_samples
    
    # --- STEP D: BACKPROPAGATION (Calculating the Gradients) ---
    # The magical simplification: derivative of Cross-Entropy + Softmax is simply (Prediction - Truth)
    dZ = (Y_pred - Y_true) / num_samples
    
    # How much did the weights and biases contribute to this error?
    dW = np.dot(X.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    
    # --- STEP E: GRADIENT DESCENT PARAMETER UPDATE ---
    # Move parameters in the opposite direction of the gradient (down the hill)
    W -= learning_rate * dW
    b -= learning_rate * db
    
    # Print progress every 20 steps
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | Total Loss: {loss:.5f}")

# =====================================================================
# 4. TESTING THE MODEL
# =====================================================================
print("\n--- TRAINING COMPLETE ---")
print(f"Optimized Weights:\n{W}")
print(f"Optimized Biases:\n{b}\n")

print("--- MAKING A PREDICTION ---")
# Let's pass a new, unseen data point that looks a lot like Item 2 [0.8, 0.2]
new_sample = np.array([[0.75, 0.21]])

# Run a quick forward pass using the trained weights
test_Z = np.dot(new_sample, W) + b
test_exp = np.exp(test_Z - np.max(test_Z))
test_probabilities = test_exp / np.sum(test_exp)

print(f"Input Features: {new_sample[0]}")
print(f"Output Probability Distribution: {test_probabilities[0]}")
print(f"Predicted Class (Highest Probability Index): {np.argmax(test_probabilities)}")