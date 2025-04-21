import torch
import tiktoken
from train_gpt2 import GPT, GPTConfig

def load_trained_model(checkpoint_path: str, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GPT(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model

enc = tiktoken.get_encoding("gpt2")  

def generate_text(model, prompt: str, max_new_tokens: int = 50):
    device = next(model.parameters()).device
    
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)  # (1, L)
    generated = input_ids

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)         # (1, T, V)
            next_token_logits = logits[:, -1, :] 
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    # 3. 解码整个序列
    output_ids = generated[0].tolist()
    return enc.decode(output_ids)           

if __name__ == "__main__":
    model = load_trained_model("log/model4/model_19072.pt")
    prompt = "Hi, I am a language model. How can I help you today?"
    output = generate_text(model, prompt, max_new_tokens=100)
    print(output)
