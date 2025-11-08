from model import GPTModel
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from data_process import create_dataloader
from matplotlib import pyplot as plt
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,        #A
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,             #B
    "qkv_bias": False
}

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_text_example(model, idx, max_new_tokens, context_size):
    #idx -> b, seq_len
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond) #[b, seq_len, 50257]
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.concat((idx, idx_next), dim=-1)
    
    return idx

def cal_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(input=logits.flatten(0,1), target=target_batch.flatten())
    return loss

def cal_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)                                    #A
    else:
        num_batches = min(num_batches, len(data_loader))   #B
    
    for i,(input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            total_loss+=loss
        else:
            break
        
    total_loss = loss / num_batches
    return total_loss

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader, model, device, eval_iter)
        val_loss = cal_loss_loader(val_loader,model,device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_example(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch,model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                generate_and_print_sample(                                                  #G
            model, tokenizer, device, start_context
        )
        
    return train_losses, val_losses, track_tokens_seen
            
            
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    # ✅ 确保所有输入都在 CPU 上
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif isinstance(x, list) and torch.is_tensor(x[0]):
            return [t.detach().cpu().item() for t in x]
        return x

    epochs_seen = to_numpy(epochs_seen)
    tokens_seen = to_numpy(tokens_seen)     # ← 关键补充！
    train_losses = to_numpy(train_losses)
    val_losses = to_numpy(val_losses)

    # 绘图
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 第二个横轴
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda')
    tokenizer = tiktoken.get_encoding("gpt2")
    file_path = 'the-verdict.txt'
    with open (file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    total_characters = len(raw_text)
    total_tokens = len(tokenizer.encode(raw_text))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    torch.manual_seed(123)
    train_ratio = 0.9
    split_idx = int(total_characters*train_ratio)
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    train_loader = create_dataloader(
        text = train_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = create_dataloader(
        text=val_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        
    model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
epochs=10
train_losses, val_losses, track_tokens_seen = train_model_simple(model,train_loader,val_loader,optimizer,device,epochs, 
                                                                 eval_freq=5,eval_iter=1,start_context="Every effort moves you",tokenizer=tokenizer)
epochs_tensor = torch.linspace(0, epochs, len(train_losses))
plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)
torch.save(model.state_dict(), f"model-{epochs}.pth")
print("训练完成，模型已保存")