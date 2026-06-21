import numpy as np

# Simulate a sequence of 3 words (e.g., "AI is powerful")
# Each word is represented by a vector of 4 numbers (Embedding Dimension = 4)
X = np.array([
    [1.0, 0.0, 2.0, 0.5],  # Word 1: "AI"
    [0.2, 1.5, 0.1, 0.0],  # Word 2: "is"
    [2.1, 0.1, 3.0, 1.2]   # Word 3: "powerful"
])

print("Input Sequence Shape (Sequence Length, Embedding Dim):", X.shape)

# Initialize Weight Matrices for Queries, Keys, and Values
np.random.seed(42)
d_k = 4 # Dimension of keys/queries

W_Q = np.random.randn(4, d_k)
W_K = np.random.randn(4, d_k)
W_V = np.random.randn(4, 4)

# ---------------------------------------------------------------------
# MANDATORY TRANSFORMER CALCULATIONS
# ---------------------------------------------------------------------

# Step 1: Project inputs into Query, Key, and Value spaces
Queries = np.dot(X, W_Q)
Keys = np.dot(X, W_K)
Values = np.dot(X, W_V)

# Step 2: Compute raw attention scores via Dot Product (Q * K^T)
raw_scores = np.dot(Queries, Keys.T)
print("\nRaw Attention Scores (Compatibility Matrix):\n", raw_scores)

# Step 3: Scale the scores to stabilize gradients
scaled_scores = raw_scores / np.sqrt(d_k)

# Step 4: Apply Softmax row-wise to create the Attention Map
shift_scores = scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
exp_scores = np.exp(shift_scores)
attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

print("\nFinal Attention Matrix (Probabilities summing to 1.0 per row):\n", attention_weights)

# Step 5: Multiply weights by Values to get context-rich output vectors
output = np.dot(attention_weights, Values)
print("\nFinal Attention Head Output Shape:", output.shape)