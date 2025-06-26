import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MiniTransformer  # Ensure your model is saved as model.py or adjust this path accordingly

# -------- Configuration --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "best_model.pt"
vocab_size = 40481  # Must match the one used during training
max_len = 1024   # Max input length used in model
max_gen_len = 50    # Max number of tokens to generate
top_k = 20          # Top-k sampling
temperature = 1.0   # Sampling temperature

# -------- Load Tokenizer and Model --------
tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  
tokenizer.pad_token = '[PAD]'

model = MiniTransformer(
    vocab_size=vocab_size,
    d_model=256,
    num_heads=8,
    d_ff=512,
    num_layers=2,
    max_len=max_len
)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# -------- Generation Function --------
def generate_answer(question_text):
    encoded = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_gen_len):
            outputs = model(generated, attention_mask)
            next_token_logits = outputs[:, -1, :] / temperature

            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(F.softmax(next_token_logits, dim=-1), top_k)
            next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, 1))

            # Append next token
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping on [EOS] token
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Extend attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype).to(device)],
                dim=1
            )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

# -------- Interactive Script --------
if __name__ == "__main__":
    print(" QA Model is ready. Type your question. Type 'quit' to exit.")
    while True:
        q = input("Question (title + description): ").strip()
        if q.lower() in ("quit", "exit"):
            break
        answer = generate_answer(q)
        print(f"\n Answer: {answer}\n")
