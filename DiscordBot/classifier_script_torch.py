"""
Asked GPT to help with: 
(1) Collecting the dataframes
(2) Making and tokenizing the dataset. 
(3) Adding weighting to the loss function of the model to help mitigate class imbalance.  
(4) Refactoring code and making it neat. 
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

# ------------------------------------------------------------
#  1) Hyperparameters and file‐paths
# ------------------------------------------------------------
YOUTUBE_CSV         = "/Users/AlfredYu/Downloads/archive/youtube_parsed_dataset.csv"
TWITTER_CSV         = "/Users/AlfredYu/Downloads/archive/twitter_parsed_dataset.csv"
ATTACK_CSV          = "/Users/AlfredYu/Downloads/archive/attack_parsed_dataset.csv"
TWITTER_RACISM_CSV  = "/Users/AlfredYu/Downloads/archive/twitter_racism_parsed_dataset.csv"
TWITTER_SEXISM_CSV  = "/Users/AlfredYu/Downloads/archive/twitter_sexism_parsed_dataset.csv"
TOXIC_CSV           = "/Users/AlfredYu/Downloads/archive/toxicity_parsed_dataset.csv"
AGGRESSION_CSV      = "/Users/AlfredYu/Downloads/archive/aggression_parsed_dataset.csv"

MAX_LENGTH    = 50      # truncate/pad each example to 50 tokens
BATCH_SIZE    = 64
EMBED_DIM     = 128
HIDDEN_DIM    = 64
NUM_CLASSES   = 2
LR            = 1e-3
NUM_EPOCHS    = 5
NUM_WORKERS   = 2      # set to 0 if you still see multiprocessing issues
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
#  2) Utility: load + merge + preprocess CSVs (with extra checks)
# ------------------------------------------------------------
def load_and_prepare():
    # List of (filepath, name) tuples for all CSVs to include
    csv_list = [
        (YOUTUBE_CSV,        "youtube_parsed_dataset.csv"),
        (TWITTER_CSV,        "twitter_parsed_dataset.csv"),
        (ATTACK_CSV,         "attack_parsed_dataset.csv"),
        (TWITTER_RACISM_CSV, "twitter_racism_parsed_dataset.csv"),
        (TWITTER_SEXISM_CSV, "twitter_sexism_parsed_dataset.csv"),
        (TOXIC_CSV,          "toxicity_parsed_dataset.csv"),
        (AGGRESSION_CSV,     "aggression_parsed_dataset.csv"),
    ]

    dfs = []
    for path, name in csv_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file: {path}")
        df = pd.read_csv(path)

        # Check for required columns
        if "Text" not in df.columns or "oh_label" not in df.columns:
            raise ValueError(
                f"Expected columns ['Text','oh_label'] in '{name}' but got {list(df.columns)}"
            )

        # Keep only the needed columns
        df = df[["Text", "oh_label"]].copy()
        dfs.append(df)

    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Drop any rows with missing Text or missing oh_label
    df = df.dropna(subset=["Text", "oh_label"]).reset_index(drop=True)

    # Attempt to coerce "oh_label" into integers 0/1.
    def map_label(x):
        # If it’s already an integer or float 0/1, accept it
        if pd.api.types.is_numeric_dtype(type(x)):
            try:
                iv = int(x)
                if iv in (0, 1):
                    return iv
            except Exception:
                return None
        # Otherwise, match strings
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl == "cyberbully":
                return 1
            if xl == "not_cyberbully":
                return 0
        return None  # unmapped -> drop

    df["label"] = df["oh_label"].map(map_label)

    # Drop any rows where map_label returned None, then cast to int
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Ensure at least one row remains
    if len(df) == 0:
        raise ValueError(
            "After filtering for valid 'label' values, no rows remain. "
            "Please check that 'oh_label' contains either integers 0/1 or strings "
            "'cyberbully' / 'not_cyberbully'."
        )

    # Keep only ["Text","label"]
    df = df[["Text", "label"]]
    return df


# ------------------------------------------------------------
#  3) Custom Dataset that tokenizes on the fly and pads/truncates
# ------------------------------------------------------------
class CyberbullyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts     = dataframe["Text"].tolist()
        self.labels    = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize (truncate if necessary)
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        # Pad on the right if shorter than max_length
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        else:
            tokens = tokens[: self.max_length]

        tokens = torch.tensor(tokens, dtype=torch.long)
        label  = torch.tensor(label, dtype=torch.long)
        return tokens, label


# ------------------------------------------------------------
#  4) Simple “Embedding → mean‐pool → two‐layer MLP” classifier
# ------------------------------------------------------------
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, hidden_two_dim, num_classes, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.act1      = nn.ReLU()
        self.fc2       = nn.Linear(hidden_dim, hidden_two_dim)
        self.act2      = nn.ReLU()
        self.fc3       = nn.Linear(hidden_two_dim, num_classes)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, max_length)
        """
        embedded = self.embedding(input_ids)  
        mask     = (input_ids != tokenizer.pad_token_id).unsqueeze(-1).float()  

        summed     = (embedded * mask).sum(dim=1)       
        lengths    = mask.sum(dim=1).clamp(min=1)       
        mean_pooled = summed / lengths                  

        x = self.fc1(mean_pooled)                       
        x = self.act1(x)
        x = self.fc2(x)                                  
        x = self.act2(x)
        logits = self.fc3(x)                             
        return logits


# ------------------------------------------------------------
#  5) Training / evaluation loops
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += input_ids.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ------------------------------------------------------------
#  6) Main entry point (multiprocessing‐safe)
# ------------------------------------------------------------
def main():
    # 6.1) Load + preprocess data (drops invalid‐label rows)
    df_all = load_and_prepare()
    train_df, val_df = train_test_split(df_all, test_size=0.1, random_state=42)

    # 6.2) Compute class weights for imbalance
    counts = train_df["label"].value_counts().sort_index()
    n_not = counts.get(0, 0)
    n_yes = counts.get(1, 0)
    total = n_not + n_yes
    w0 = total / (2 * n_not) if n_not > 0 else 1.0
    w1 = total / (2 * n_yes) if n_yes > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)
    print(f"Computed class weights: {class_weights.tolist()}")

    # 6.3) Initialize tokenizer (we only need vocab + pad token)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pad_id = tokenizer.pad_token_id

    # 6.4) Create Datasets
    train_dataset = CyberbullyDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = CyberbullyDataset(val_df, tokenizer, MAX_LENGTH)

    # 6.5) WeightedRandomSampler to oversample minority class
    labels = train_df["label"].values
    label_counts = [n_not, n_yes]
    weight_per_class = [1.0 / c if c > 0 else 0.0 for c in label_counts]
    sample_weights = torch.tensor([weight_per_class[l] for l in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # 6.6) Create DataLoaders (with sampler instead of shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,       # oversample minority class each epoch
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 6.7) Instantiate model + weighted cross‐entropy loss + optimizer
    model = SimpleTextClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        hidden_two_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        padding_idx=pad_id,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6.8) Training loop
    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}"
        )

        # Optionally save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "simple_text_classifier_best.pt")

    print(f"\nFinished training. Best validation acc: {best_val_acc:.4f}")


# Only call main() if this script is run directly
if __name__ == "__main__":
    main()
