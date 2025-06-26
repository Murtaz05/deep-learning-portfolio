
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils import *
from model import *

vocab_size = len(tokenizer)
model = MiniTransformer(vocab_size=vocab_size).to(device)

criterion = CrossEntropyLoss(ignore_index=-100)  # ignore padding
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=10)


# %%

def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pt'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

def calculate_perplexity(loss):
    return torch.exp(loss)


# %%
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        
        # Reshape for loss computation
        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, attention_mask)
            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            loss = criterion(output, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()



# %%
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict

# Helper function for generation
def greedy_decode(model, input_ids, attention_mask, tokenizer, max_length=50, device='cpu'):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        generated = input_ids

        for _ in range(max_length):
            outputs = model(generated, attention_mask)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat((generated, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
        return generated

# Helper function to decode token IDs to strings
def decode_predictions(pred_ids, tokenizer):
    return [tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in pred_ids]

# BLEU score calculation
def calculate_bleu(model, val_loader, tokenizer, device, max_length=50):
    model.eval()
    bleu_scores = []
    smooth_fn = SmoothingFunction().method4

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            generated_ids = greedy_decode(model, input_ids, attention_mask, tokenizer, max_length=max_length, device=device)
            decoded_preds = decode_predictions(generated_ids, tokenizer)

            # Convert labels to strings for BLEU scoring
            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id  # replace -100 with pad_token_id
            decoded_labels = decode_predictions(label_ids, tokenizer)

            for ref, hyp in zip(decoded_labels, decoded_preds):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_fn)
                bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores)

def train_model_with_metrics(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    tokenizer,
    device, 
    epochs=10, 
    patience=3, 
    checkpoint_path='best_model.pt',
    max_gen_len=50
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = defaultdict(list)

    for epoch in range(epochs):
        train_loss, train_ppl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        bleu = calculate_bleu(model, val_loader, tokenizer, device, max_length=max_gen_len)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)
        history['bleu'].append(bleu)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f} | BLEU: {bleu:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print("  Validation loss decreased, saving model...")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            print(f"  No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return history


# %%
import torch
torch.cuda.empty_cache()


# %%
history = train_model_with_metrics(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    tokenizer=tokenizer,
    device=device,
    epochs=10,
    patience=3,
    checkpoint_path='best_model.pt',
    max_gen_len=50  # can be adjusted
)


# %% [markdown]
# Visualization

# %%
import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Loss
    axs[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axs[0].plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
    axs[0].set_title('Loss per Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Perplexity
    axs[1].plot(epochs, history['train_ppl'], label='Train Perplexity', marker='o')
    axs[1].plot(epochs, history['val_ppl'], label='Validation Perplexity', marker='o')
    axs[1].set_title('Perplexity per Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Perplexity')
    axs[1].legend()
    axs[1].grid(True)

    # Plot BLEU score
    axs[2].plot(epochs, history['bleu'], label='BLEU Score', marker='o', color='green')
    axs[2].set_title('BLEU Score per Epoch')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('BLEU Score')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# %%
plot_training_history(history)
