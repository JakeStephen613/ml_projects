import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# For reproducibility: this makes random numbers repeatable
torch.manual_seed(42)
NUM_SAMPLES = 200
BATCH_SIZE = 32
NUM_EPOCHS = 200 ## num times run through data
LEARNING_RATE = 0.1 ## how much you want parameters to update, too high overshoots, too low doesnt learn


## creation of syntethic dataset. I want it to learn relationship y = 3x^2 - 5x + 8 + noise. 

# True underlying relationship (this is what our model should discover)
true_weight = 3.0
true_weight_2 = 5.0
true_bias = 8.0

# Small random noise to make it more realistic
noise = 0.5 * torch.randn(NUM_SAMPLES, 1)

# shape: (NUM_SAMPLES, 1) -> 1 feature per sample
x = torch.randn(NUM_SAMPLES, 1)
x2 = x ** 2
X = torch.cat([x, x2], dim=1)   
y = true_weight_1 * x2 + true_weight_2 * x + true_bias + noise

# 4. WRAP DATA IN A DATASET AND DATALOADER
# TensorDataset stores inputs and targets together.
dataset = TensorDataset(X, y)

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
class NonlinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            ## first hidden layer
            nn.Linear(32, 32),
            nn.ReLU(),
            ## second hidden layer
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(X)

# Instantiate the model
model = NonlinearModel()

# 6. DEFINE LOSS FUNCTION + OPTIMIZER
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
# 8. INSPECT LEARNED PARAMETERS (OPTIONAL)
# ============================
# For a nonlinear network (with hidden layers), individual weights
# are not as interpretable as in simple linear regression.
# But we can still inspect their shapes or a summary:

print("\n=== Model parameter summary ===")
for name, param in model.named_parameters():
    print(f"{name:20s} shape: {tuple(param.shape)}")


# ============================
# 9. TEST THE MODEL ON NEW DATA
# ============================
# Let's predict y for a few new x values and compare to the true *quadratic* formula
# (without noise) for reference.
# We'll use torch.no_grad() to tell PyTorch we don't need gradients here.

model.eval()  # put the model in 'evaluation' mode (important for some layers)

with torch.no_grad():
    # Some new x points
    test_x = torch.tensor([[-2.0], [0.0], [1.0], [2.0]])

    # Model predictions
    pred_y = model(test_x)

    # True y (without noise) for comparison, using the quadratic relationship:
    # y = true_weight_1 * x^2 + true_weight_2 * x + true_bias
    true_y = (
        true_weight_1 * (test_x ** 2)
        + true_weight_2 * test_x
        + true_bias
    )

print("\n=== Predictions on new data ===")
for i in range(len(test_x)):
    x_val = test_x[i].item()
    pred_val = pred_y[i].item()
    true_val = true_y[i].item()
    print(f"x = {x_val:5.2f} | model y ≈ {pred_val:8.4f} | true y = {true_val:8.4f}")
