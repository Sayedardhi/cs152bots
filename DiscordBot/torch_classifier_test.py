import torch
from transformers import AutoTokenizer
import torch.nn as nn
"""
Asked GPT to define the logic for making calls."


"""
# ─────────────────────────────────────────────────────────────────────────────
# 1) Re‐define (or import) your model class exactly as in training
# ─────────────────────────────────────────────────────────────────────────────
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, hidden_dim_two, num_classes, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.act1      = nn.ReLU()
        self.fc2       = nn.Linear(hidden_dim, hidden_dim_two)
        self.act2      = nn.ReLU()
        self.fc3       = nn.Linear(hidden_dim_two, num_classes)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, max_length)
        """
        embedded = self.embedding(input_ids)  # (batch_size, max_length, embed_dim)
        mask     = (input_ids != tokenizer.pad_token_id).unsqueeze(-1).float()  # (batch_size, max_length, 1)

        summed = (embedded * mask).sum(dim=1)   # (batch_size, embed_dim)
        lengths = mask.sum(dim=1).clamp(min=1)   # (batch_size, 1)
        mean_pooled = summed / lengths           # (batch_size, embed_dim)

        x = self.fc1(mean_pooled)                # (batch_size, hidden_dim)
        x = self.act1(x)
        x = self.fc2(x)                           # (batch_size, hidden_two_dim)
        x = self.act2(x)
        logits = self.fc3(x)                      # (batch_size, num_classes)
        return logits

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load tokenizer & model weights
# ─────────────────────────────────────────────────────────────────────────────
TOKENIZER_NAME = "bert-base-uncased"
MODEL_PATH     = "simple_text_classifier_best.pt"   # path to your .pt file
MAX_LENGTH     = 50
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a) Load the same tokenizer you used during training
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
vocab_size = tokenizer.vocab_size
pad_id     = tokenizer.pad_token_id

# b) Instantiate the exact same architecture
model = SimpleTextClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=64,
    hidden_dim_two=64,
    num_classes=2,
    padding_idx=pad_id,
).to(DEVICE)

# c) Load the saved weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 3) Single‐sentence inference function (takes only `message` as input)
# ─────────────────────────────────────────────────────────────────────────────
def predict_message(message: str) -> (str, torch.Tensor):
    """
    Given a raw text `message`, returns:
      - predicted label ("not_cyberbully" or "cyberbully")
      - tensor of class probabilities [prob_not_cyberbully, prob_cyberbully]
    """
    # 1) Tokenize & convert to IDs (truncation + padding)
    tokens = tokenizer.encode(
        message,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
    )
    if len(tokens) < MAX_LENGTH:
        tokens = tokens + [tokenizer.pad_token_id] * (MAX_LENGTH - len(tokens))
    else:
        tokens = tokens[:MAX_LENGTH]

    # 2) Build a (1, MAX_LENGTH) tensor of input IDs
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 3) Forward pass through the model
    with torch.no_grad():
        logits = model(input_ids)                    # shape: (1, num_classes)
        probs  = torch.softmax(logits, dim=1).squeeze(0)  # shape: (num_classes,)

    # 4) Map prediction index to human-readable label
    pred_idx = torch.argmax(probs).item()
    label_map = {0: "not_cyberbully", 1: "cyberbully"}
    predicted_label = label_map[pred_idx]

    return predicted_label, probs.cpu()

# ─────────────────────────────────────────────────────────────────────────────
# 4) Example usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_message = "I will eat you."
    label, class_probs = predict_message(test_message)
    print(f"Message: {test_message}")
    print(f"Predicted label: {label}")
    print(f"Class probabilities (not_cyberbully vs. cyberbully): {class_probs.numpy()}")
