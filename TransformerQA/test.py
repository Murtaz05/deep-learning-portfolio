
# %% [markdown]
# Testing

# %%
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
import torch
from tqdm import tqdm
from data_utils import *
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_answer(model, input_ids, attention_mask, tokenizer, device, max_length=50):
    model.eval()
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids, attention_mask)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_ids.append(next_token.item())

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def evaluate_test_set(model, test_loader, tokenizer, device, max_gen_len=50):
    model.eval()
    predictions = []
    references = []
    results = []

    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']  

        for i in range(input_ids.size(0)):
            pred = generate_answer(model, input_ids[i], attention_mask[i], tokenizer, device, max_length=max_gen_len)
            true = tokenizer.decode([id for id in labels[i].tolist() if id != -100 and id != tokenizer.pad_token_id], skip_special_tokens=True)

            predictions.append(pred.split())
            references.append([true.split()])
            results.append((pred, true))

    bleu = corpus_bleu(references, predictions)
    print(f"Test BLEU Score: {bleu:.4f}")
    return results, bleu


# %%
# Load best model
model = MiniTransformer(
    vocab_size=40481,
    d_model=256,
    num_heads=8,
    d_ff=512,
    num_layers=2,
    max_len=512
)
model.load_state_dict(torch.load("best_model.pt"))
model.to(device)

# Run test
results, bleu_score = evaluate_test_set(model, test_loader, tokenizer, device)


# %%



