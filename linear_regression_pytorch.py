# ===========
# 1. IMPORTS
# ===========
# torch: main PyTorch library (tensors, autograd, etc.)
import torch
# nn: neural network layers and loss functions
import torch.nn as nn
# optim: optimizers (SGD, Adam, etc.)
import torch.optim as optim
# Data utilities for mini-batches
from torch.utils.data import TensorDataset, DataLoader

# For reproducibility: this makes random numbers repeatable
torch.manual_seed(42)

# =======================================
# 2. HYPERPARAMETERS / CONFIG VARIABLES
# =======================================
# Number of samples in our synthetic dataset
NUM_SAMPLES = 200

# Batch size for training (how many samples per gradient step)
BATCH_SIZE = 32

# How many times we iterate over the entire dataset
NUM_EPOCHS = 200

# Learning rate for the optimizer (step size in parameter space)
LEARNING_RATE = 0.1


# ================================
# 3. CREATE A SYNTHETIC DATASET
# ================================
# We want to learn the relationship y = 3x + 2 + noise
# We'll generate x values randomly and compute y from them.

# Generate random x values from a normal distribution
# shape: (NUM_SAMPLES, 1) -> 1 feature per sample
x = torch.randn(NUM_SAMPLES, 1)

# True underlying relationship (this is what our model should discover)
true_weight = 3.0
true_bias = 2.0

# Small random noise to make it more realistic
noise = 0.5 * torch.randn(NUM_SAMPLES, 1)

# Compute y according to the formula
y = true_weight * x + true_bias + noise

# At this point:
#   x: input tensor with shape [NUM_SAMPLES, 1]
#   y: target tensor with shape [NUM_SAMPLES, 1]


# ==========================================
# 4. WRAP DATA IN A DATASET AND DATALOADER
# ==========================================
# TensorDataset stores inputs and targets together.
dataset = TensorDataset(x, y)

# DataLoader handles batching and (optionally) shuffling.
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True  # shuffle each epoch so the model sees data in random order
)


# ===============================
# 5. DEFINE THE MODEL (nn.Module)
# ===============================
# We’ll build a tiny neural network with just a single Linear layer:
#     input_dim = 1  -> one feature (x)
#     output_dim = 1 -> one prediction (y_hat)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()  # initialize parent nn.Module
        # nn.Linear(in_features, out_features)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        """
        Forward pass: defines how input x is transformed to output.
        x: tensor of shape [batch_size, 1]
        returns: predictions of shape [batch_size, 1]
        """
        return self.linear(x)


# Instantiate the model
model = LinearRegressionModel()

# ====================================
# 6. DEFINE LOSS FUNCTION + OPTIMIZER
# ====================================
# Mean Squared Error (MSE) is standard for regression
criterion = nn.MSELoss()

# We’ll use Stochastic Gradient Descent (SGD) to update model parameters.
optimizer = optim.SGD(
    model.parameters(),   # which parameters to update (the linear layer’s weights and bias)
    lr=LEARNING_RATE      # learning rate
)


# ============================
# 7. TRAINING LOOP
# ============================
# A typical training loop in PyTorch looks like:
#   for each epoch:
#       for each batch:
#           1. zero gradients
#           2. forward pass
#           3. compute loss
#           4. backward pass (loss.backward())
#           5. optimizer.step() to update params

for epoch in range(NUM_EPOCHS):
    # Keep track of the average loss across all batches for this epoch
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        # 1. Zero the gradients from the previous step
        optimizer.zero_grad()

        # 2. Forward pass: get predictions from the model
        #    shape: [batch_size, 1]
        preds = model(batch_x)

        # 3. Compute loss between predictions and true targets
        loss = criterion(preds, batch_y)

        # 4. Backward pass: compute gradients of loss wrt model params
        loss.backward()

        # 5. Update parameters using the gradients
        optimizer.step()

        # Accumulate batch loss (detach() so we don't keep graph)
        epoch_loss += loss.detach().item()

    # Compute average loss over all batches in this epoch
    avg_epoch_loss = epoch_loss / len(train_loader)

    # Print training progress every 20 epochs
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS}, Loss: {avg_epoch_loss:.4f}")


# ============================
# 8. INSPECT LEARNED PARAMETERS
# ============================
# After training, the model's linear layer should have weight ~3 and bias ~2
learned_weight = model.linear.weight.item()  # weight is a 1x1 tensor
learned_bias = model.linear.bias.item()      # bias is a scalar tensor

print("\n=== Learned parameters ===")
print(f"True weight:   {true_weight:.3f}, Learned weight: {learned_weight:.3f}")
print(f"True bias:     {true_bias:.3f}, Learned bias:   {learned_bias:.3f}")

# ============================
# 9. TEST THE MODEL ON NEW DATA
# ============================
# Let's predict y for a few new x values and compare to the true formula.
# We'll use torch.no_grad() to tell PyTorch we don't need gradients here.

model.eval()  # put the model in 'evaluation' mode (important when using layers like dropout/batchnorm)

with torch.no_grad():
    # Some new x points
    test_x = torch.tensor([[-2.0], [0.0], [1.0], [2.0]])

    # Model predictions
    pred_y = model(test_x)

    # True y (without noise) for comparison
    true_y = true_weight * test_x + true_bias

print("\n=== Predictions on new data ===")
for i in range(len(test_x)):
    x_val = test_x[i].item()
    pred_val = pred_y[i].item()
    true_val = true_y[i].item()
    print(f"x = {x_val:5.2f} | model y ≈ {pred_val:6.3f} | true y = {true_val:6.3f}")
