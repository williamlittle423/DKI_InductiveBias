import os
import copy
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint


# ---------- Define Models --------------

# cNODE2 from Wang et al.
class OriginalCNODE(torch.nn.Module):
    def __init__(self, n_species):
        super().__init__()
        self.fcc1 = torch.nn.Linear(n_species, n_species)
        self.fcc2 = torch.nn.Linear(n_species, n_species)

    def forward(self, t, y):
        out = self.fcc1(y)
        out = self.fcc2(out)

        mean_interaction = torch.sum(y * out, dim=1, keepdim=True)

        return y * (out - mean_interaction)
    
# Linear Model
class LinearAbundanceModel(torch.nn.Module):
    def __init__(self, n_species):
        super().__init__()
        self.linear = torch.nn.Linear(n_species, n_species)

    def forward(self, z):
        logits = self.linear(z)
        p_pred = torch.softmax(logits, dim=1)
        return p_pred

# Neural ODE Model
class NeuralODE(torch.nn.Module):
    def __init__(self, n_species):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_species, n_species),
            torch.nn.Tanh(),
            torch.nn.Linear(n_species, n_species),
        )

    def forward(self, t, y):
        out = self.net(y)

        mean_interaction = torch.sum(y * out, dim=1, keepdim=True)

        dydt = y * (out - mean_interaction)

        return dydt

class ResidualMLPAbundanceModel(torch.nn.Module):
    def __init__(self, n_species, hidden_dim=64):
        super().__init__()

        self.input = torch.nn.Linear(n_species, hidden_dim)

        self.block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.output = torch.nn.Linear(hidden_dim, n_species)

    def forward(self, z):
        h = self.input(z)
        h = h + self.block(h)
        logits = self.output(h)
        return torch.softmax(logits, dim=1)

# MLP
class MLPAbundanceModel(torch.nn.Module):
    def __init__(self, n_species, hidden_dim=64):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_species, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_species),
        )

    def forward(self, z):
        logits = self.net(z)
        return torch.softmax(logits, dim=1)

class AttentionAbundanceModel(torch.nn.Module):
    def __init__(self, n_species, embed_dim=16, num_heads=2):
        super().__init__()

        self.n_species = n_species

        # One learned embedding per species
        self.species_embedding = torch.nn.Embedding(n_species, embed_dim)

        # Converts binary presence/absence into embedding scale
        self.presence_projection = torch.nn.Linear(1, embed_dim)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, 1),
        )

    def forward(self, z):
        """
        z shape: batch_size x n_species
        output shape: batch_size x n_species
        """

        batch_size = z.size(0)

        species_ids = torch.arange(
            self.n_species,
            device=z.device
        )

        species_emb = self.species_embedding(species_ids)
        species_emb = species_emb.unsqueeze(0).expand(batch_size, -1, -1)

        presence_emb = self.presence_projection(z.unsqueeze(-1))

        x = species_emb + presence_emb

        # Mask absent species so they do not receive attention
        key_padding_mask = z <= 0

        attn_out, attn_weights = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        logits = self.output(attn_out).squeeze(-1)

        # Force absent species to have zero abundance
        logits = logits.masked_fill(z <= 0, -1e9)

        p_pred = torch.softmax(logits, dim=1)

        return p_pred

def integrate_fixed_time(func, y0, dt=0.01, steps=100):
    """
    Fixed-time integration for cNODE-style models.
    Mimics the original code's fixed trajectory behavior,
    but avoids torchdiffeq.odeint.
    """
    y = y0
    t = torch.tensor(0.0, device=y0.device)

    for _ in range(steps):
        y = y + dt * func(t, y)

        # keep numerical stability
        y = torch.clamp(y, min=0.0)
        y = y / (y.sum(dim=1, keepdim=True) + 1e-8)

        t = t + dt

    return y

def integrate_until_equilibrium(func, y0, dt=0.05, max_steps=1000, tol=1e-6):
    y = y0
    t = torch.tensor(0.0, device=y0.device)

    for step in range(max_steps):
        y_next = y + dt * func(t, y)

        y_next = torch.clamp(y_next, min=0.0)
        y_next = y_next / (y_next.sum(dim=1, keepdim=True) + 1e-8)

        change = torch.norm(y_next - y, dim=1).mean()

        y = y_next
        t = t + dt

        if change < tol:
            break

    return y

CNODE_STEPS = 50
MAX_EQ_STEPS = 200
EQ_TOL = 1e-4

CNODE_DT = 0.01

def forward_model(model, model_type, batch_z):
    model_type = model_type.lower()

    if model_type == "cnode":
        return integrate_fixed_time(
            model,
            batch_z,
            dt=CNODE_DT,
            steps=CNODE_STEPS,
        )

    elif model_type == "node":
        return integrate_until_equilibrium(
            model,
            batch_z,
            dt=DT,
            max_steps=MAX_EQ_STEPS,
            tol=EQ_TOL,
        )
    elif model_type in ["linear", "mlp", "residualmlp", "attention"]:
        return model(batch_z)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -------------------------
# Config
# -------------------------
MAX_EPOCHS = 500
LR = 1e-2
BATCH_SIZE = 256

DT = 0.05
ODE_STEPS = 200

ALPHA_EQ = 0.01

TRAIN_PATH = "data/Ptrain.csv"
TEST_PATH = "data/Ptest.csv"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# -------------------------
# Data processing
# -------------------------
def process_data(P):
    """
    P: species x samples relative abundance matrix.

    Returns:
        P: samples x species abundance tensor
        Z: samples x species binary mask tensor
    """
    Z = P.copy()
    Z[Z > 0] = 1.0

    P = P / (P.sum(axis=0, keepdims=True) + 1e-8)
    Z = Z / (Z.sum(axis=0, keepdims=True) + 1e-8)

    P = torch.tensor(P.T, dtype=torch.float32)
    Z = torch.tensor(Z.T, dtype=torch.float32)

    return P, Z


# -------------------------
# Losses
# -------------------------
# Bray-Curtis Dissimilarity
def loss_bc(p_pred, p_true, eps=1e-8):
    numerator = torch.sum(torch.abs(p_pred - p_true))
    denominator = torch.sum(torch.abs(p_pred + p_true)) + eps
    return numerator / denominator

# for Neural ODE
def equilibrium_loss(func, p_pred):
    t0 = torch.tensor(0.0, device=p_pred.device)
    dpdt = func(t0, p_pred)
    return torch.mean(dpdt ** 2)


# -------------------------
# Train / validate
# -------------------------
def train_model(
    model,
    model_type,
    max_epochs,
    batch_size,
    lr,
    z_train,
    p_train,
    z_val,
    p_val,
    z_test,
    z_all,
):
    model_type = model_type.lower()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(z_train, p_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        TensorDataset(z_val, p_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_state_dict = None

    train_losses = []
    val_losses = []

    print(f"Beginning training for model type: {model_type}")

    for epoch in range(max_epochs):
        model.train()

        train_total = 0.0
        n_train = 0

        for batch_z, batch_p_true in train_loader:
            optimizer.zero_grad(set_to_none=True)

            p_pred = forward_model(
                model=model,
                model_type=model_type,
                batch_z=batch_z,
            )

            abundance_loss = loss_bc(p_pred, batch_p_true)

            if model_type == "node":
                eq_loss = equilibrium_loss(model, p_pred)
                loss = abundance_loss + ALPHA_EQ * eq_loss
            elif model_type in ["linear", "mlp", "residualmlp", "attention", "cnode"]:
                loss = abundance_loss
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            loss.backward()
            optimizer.step()

            bs = batch_z.size(0)
            train_total += loss.item() * bs
            n_train += bs

        avg_train_loss = train_total / n_train
        train_losses.append(avg_train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_total = 0.0
        n_val = 0

        with torch.inference_mode():
            for batch_z, batch_p_true in val_loader:
                p_pred = forward_model(
                    model=model,
                    model_type=model_type,
                    batch_z=batch_z,
                )

                val_loss = loss_bc(p_pred, batch_p_true)

                bs = batch_z.size(0)
                val_total += val_loss.item() * bs
                n_val += bs

        avg_val_loss = val_total / n_val
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

            save_path = os.path.join(
                RESULTS_DIR,
                f"best_{model_type}_model.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_type": model_type,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": lr,
                    "batch_size": batch_size,
                },
                save_path,
            )

            print(f"Saved new best {model_type} model at epoch {epoch} with loss {best_val_loss:.5f}")

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"epoch={epoch:03d}, "
                f"model={model_type}, "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_loss={avg_val_loss:.6f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()

    def predict_in_batches(z_data):
        loader = DataLoader(
            TensorDataset(z_data),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        preds = []

        with torch.inference_mode():
            for (batch_z,) in loader:
                p_pred = forward_model(
                    model=model,
                    model_type=model_type,
                    batch_z=batch_z,
                )

                preds.append(p_pred.cpu())

        return torch.cat(preds, dim=0).numpy()

    q_test = predict_in_batches(z_test)
    q_train_all = predict_in_batches(z_all)

    return train_losses, val_losses, q_test, q_train_all, model

P_train_full = np.loadtxt(TRAIN_PATH, delimiter=",")

BATCH_SIZE_ORIGINAL_CNODE = 256

BATCH_TIME = 100

T_ORIGINAL = torch.arange(
    0.0,
    BATCH_TIME,
    0.01,
    device=device,
)[:BATCH_TIME]

n_cols = P_train_full.shape[1]
val_size = int(0.2 * n_cols)

val_indices = np.random.choice(
    n_cols,
    size=val_size,
    replace=False,
)

train_indices = np.setdiff1d(np.arange(n_cols), val_indices)

P_val_np = P_train_full[:, val_indices]
P_train_np = P_train_full[:, train_indices]

p_train, z_train = process_data(P_train_np)
p_val, z_val = process_data(P_val_np)
p_all, z_all = process_data(P_train_full)

P_test_np = np.loadtxt(TEST_PATH, delimiter=",")
p_test, z_test = process_data(P_test_np)

n_species = p_train.shape[1]

print(f"Training samples: {p_train.shape[0]}")
print(f"Validation samples: {p_val.shape[0]}")
print(f"Test samples: {p_test.shape[0]}")
print(f"Species/features: {n_species}")

# Move all tensors to GPU once
p_train = p_train.to(device)
z_train = z_train.to(device)

p_val = p_val.to(device)
z_val = z_val.to(device)

z_test = z_test.to(device)
z_all = z_all.to(device)

 # cNODE model
models = {
    "cnode": OriginalCNODE(n_species),
    "node": NeuralODE(n_species),
    "linear": LinearAbundanceModel(n_species),
    "residualmlp": ResidualMLPAbundanceModel(n_species, hidden_dim=512),
    "attention": AttentionAbundanceModel(n_species),
}

model_epochs = {"cnode": 300, 
                "node": 1000,
          "linear": 5000, 
          "residualmlp": 5000,
          "attention": 5000}


model_lr = {"cnode": 0.01,
            "node": 0.01, 
          "linear": 0.1, 
          "residualmlp": 0.01,
          "attention": 0.01}

for key in models.keys():

    batch_size = BATCH_SIZE_ORIGINAL_CNODE if key == "cnode" else BATCH_SIZE

    train_losses, val_losses, q_test, q_train_all, trained_model = train_model(
        model=models[key],
        model_type=key,
        max_epochs=model_epochs[key],
        batch_size=batch_size,
        lr=model_lr[key],
        z_train=z_train,
        p_train=p_train,
        z_val=z_val,
        p_val=p_val,
        z_test=z_test,
        z_all=z_all,
    )

    np.savetxt(
        os.path.join(RESULTS_DIR, f"qtst_{key}.csv"),
        q_test,
        delimiter=",",
    )

    np.savetxt(
        os.path.join(RESULTS_DIR, f"qtrn_{key}.csv"),
        q_train_all,
        delimiter=",",
    )

    np.savetxt(
        os.path.join(RESULTS_DIR, f"loss_train_{key}.csv"),
        np.array(train_losses),
        delimiter=",",
    )

    np.savetxt(
        os.path.join(RESULTS_DIR, f"loss_val_{key}.csv"),
        np.array(val_losses),
        delimiter=",",
    )

print("Done.")
print(f"Saved predictions and losses to: {RESULTS_DIR}")
