import torch
import torch.nn as nn

vocab_size = 10    # Smaller vocabulary for better visualization
emb_dim = 4       # Smaller embedding dimension
token_idx = 3     # Index of the token we want to embed

# Approach 1: Using nn.Embedding which directly maps indices to dense vectors
embedding = nn.Embedding(vocab_size, emb_dim)

# Approach 2: Using nn.Linear to achieve the same effect
# The Linear layer performs: output = input @ weight.t() + bias
# In our case, bias=False so: output = input @ weight.t()
linear = nn.Linear(vocab_size, emb_dim, bias=False)

# Copy and transpose embedding weights for the linear layer
# Embedding weights shape: (vocab_size, emb_dim)
# Linear weights shape: (emb_dim, vocab_size)  <- notice the transpose
linear.weight.data = embedding.weight.data.t()

# Create one-hot input for linear layer - zeros everywhere except position token_idx
one_hot = torch.zeros(vocab_size)
one_hot[token_idx] = 1

# Get embedding by directly indexing into the embedding matrix
# embedding.weight[token_idx] is what happens under the hood
emb_output = embedding(torch.tensor([token_idx]))

# For linear layer: add batch dimension since linear expects shape (batch_size, input_dim)
# Result will be: one_hot @ weight.t(), which selects the token_idx row of weight.t()
linear_output = linear(one_hot.unsqueeze(0))

# Print outputs and comparison to see the equivalence
print(f"Embedding output:\n{emb_output}\n")
print(f"Linear output:\n{linear_output}\n")
print(f"Are tensors equal? {torch.equal(emb_output, linear_output)}")
print(f"Are tensors close? {torch.allclose(emb_output, linear_output)}")

# Verify outputs are the same within numerical precision
# Uses default tolerances: rtol=1e-5, atol=1e-8
# Formula: |x - y| ≤ atol + rtol * |y|
print(f"\nMaximum difference between tensors: {(emb_output - linear_output).abs().max().item()}")