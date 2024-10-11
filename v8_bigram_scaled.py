import os
import torch
import onnx
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
torch.manual_seed(1337)  # for reproducibility

SEP = 50 * '-'

# hyperparameters ----------------------------------------------------------------------------------
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256  # what i sthe maximum context length for predictions
max_iters = 5000  # how many iterations to train for
eval_interval = 500  # how often to evaluate the model
learning_rate = 3e-4  # how fast we update the weights, lowering the learning rate as the model gets bigger
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # check if GPU is available
eval_iters = 200  # how many batches to average for evaluation
n_embd = 384  # number of embedding dimensions
n_head = 6  # number of self-attention heads
n_layer = 6  # number of transformer blocks
dropout = 0.2  # dropout rate

# dataset ------------------------------------------------------------------------------------------
dataset_path = 'dataset/tiny-lafontaine.txt'
with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}  # chars -> ints table
itos = {i: ch for i, ch in enumerate(chars)}  # ints -> chars table
encode = lambda s: [stoi[c] for c in s]  # encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: takes a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))   # first 90% of the data will be the training set, rest will be the validation set
train_data = data[:n]
val_data = data[n:]


# data loading -------------------------------------------------------------------------------------
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data  # choose the split
    ix = torch.randint(len(data) - block_size, (batch_size,))  # sample random starting indices for the sequences
    x = torch.stack([data[i: i + block_size] for i in ix])  # create a batch of context windows
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # create a batch of targets, one step forward
    x, y = x.to(device), y.to(device)  # move the data to the device
    return x, y


@torch.no_grad()  # this is just to reduce memory consumption, block won't call backward, no back-propagation
def estimate_loss():
    out = {}  # store the losses for the train and val splits
    model.eval()  # switch to evaluation mode
    for split in ['train', 'val']:  # iterate over both splits
        losses = torch.zeros(eval_iters)  # store the loss for each batch
        for k in range(eval_iters):  # iterate over the number of batches
            X, Y = get_batch(split)  # get a batch of data
            _, loss = model(X, Y)  # compute the loss
            losses[k] = loss.item()  # store the loss
        out[split] = losses.mean()  # store the average loss for the split
    model.train()  # switch back to training mode
    return out  # return the losses


# self-attention head ------------------------------------------------------------------------------
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # key projection
        self.query = nn.Linear(n_embd, head_size, bias=False)  # query projection
        self.value = nn.Linear(n_embd, head_size, bias=False)  # value projection
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # causal mask
        self.dropout = nn.Dropout(dropout)  # dropout layer

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # apply dropout
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# multi-attention head -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # create n_heads heads
        self.proj = nn.Linear(n_embd, n_embd)  # linear projection to get back to the original dimension

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate the outputs of each head
        out = self.proj(out)  # linear projection to get back to the original dimension
        return out


# feedforward block --------------------------------------------------------------------------------
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()  # call the constructor of the parent class
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # linear layer
            nn.ReLU(),  # activation function
            nn.Linear(4 * n_embd, n_embd),  # projection layer to get back to the original dimension
            nn.Dropout(dropout),  # dropout layer
        )

    def forward(self, x):
        return self.net(x)  # apply the feedforward block


# transformer block --------------------------------------------------------------------------------
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head  # size of the self-attention heads
        self.sa = MultiHeadAttention(n_head, head_size)  # self-attention layer
        self.ffwd = FeedForward(n_embd)  # feedforward block
        self.ln1 = nn.LayerNorm(n_embd)  # layer normalization
        self.ln2 = nn.LayerNorm(n_embd)  # layer normalization

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # apply the self-attention block. Layer normalization is applied before
        x = x + self.ffwd(self.ln2(x))  # apply the feedforward block. Layer normalization is applied before
        return x


# simple bigram model ------------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits from the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # positional embeddings
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # stack of transformer blocks
        self.ln_f = nn.LayerNorm(n_embd),  # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # output layer

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) = Batch, Time (block_size), Channels (vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # apply the transformer blocks, multiple layers of self-attention and feedforward, (B, T, C)
        logits = self.lm_head(x)  # decoder head (B, T, vocab_size)

        if targets is None:  # if we don't have targets, we can't compute the loss
            loss = None

        else:
            # reshape the logits to be (B*T, C) and the targets to be (B*T) so we can compute the loss
            B, T, C = logits.shape  # unpack batch, time, channels
            logits = logits.view(B * T, C)  # flatten the Time and Batch dimensions
            targets = targets.view(B * T)  # flatten the Time and Batch dimensions

            # compute the loss using cross entropy = quality of the logicts in respect to the targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]  # (B, T)
            # get the predictions
            logits, loss = self(idx_cond)  # (B, T, C)  internally calls the forward method in pytorch
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# train model --------------------------------------------------------------------------------------
def train_model():
    # create the model and optimizer
    model = BigramLanguageModel()
    m = model.to(device)  # move the model to the device (cuda)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)  # AdamW is a good optimizer for transformers

    # training loop ------------------------------------------------------------------------------------
    for iter in range(max_iters):

        # every once in a while evaluate the loss on the train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        _, loss = m(xb, yb)  # calling the model and passing in the input and the targets
        optimizer.zero_grad(set_to_none=True)  # clear previous gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # update the weights

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # initialize context to be a single token
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))  # generate 100 new tokens

    # save model
    save_model(model)

    return m


# save model ---------------------------------------------------------------------------------------
def save_model(model, save_path=None):
    try:
        if save_path is None:
            filename = os.path.splitext(os.path.basename(__file__))[0]
            timestamp = datetime.now().strftime('%y%m%d_%H%M')
            save_path = f'{filename}_{timestamp}.pth'

        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}.")
        return save_path

    except Exception as e:
        print(f"Error saving the model: {e}")


# load model ---------------------------------------------------------------------------------------
def load_model(model_path):
    try:
        # Load the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = BigramLanguageModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {model_path}.")
        return model

    except Exception as e:
        print(f"Error loading the model: {e}")


# run inference ------------------------------------------------------------------------------------
def run_inference(model, max_tokens=500):
    # Set to evaluation mode
    model.eval()
    # Define a starting context and run inference
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initialize with a single token
    generated_sequence = model.generate(context, max_tokens)  # Generate text
    generated_text = decode(generated_sequence[0].tolist())  # Decode the generated indices to text
    return generated_text


# export model to onnx format ----------------------------------------------------------------------
def export_onnx_model(pt_model, onnx_path):
    try:
        # Dummy input tensor of the same shape as your training input
        dummy_input = torch.zeros((1, 256), dtype=torch.long).to(device)  # Example input shape

        # Export the model to ONNX format
        torch.onnx.export(
            pt_model,  # your trained model
            dummy_input,  # example input tensor
            onnx_path,  # output file path
            input_names=["input"],  # input layer names
            output_names=["output"],  # output layer names
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # dynamic axis support
            opset_version=13  # compatibility with latest ONNX version
        )

        print(f"Model exported to {onnx_path}.")

    except Exception as e:
        print(f"Error exporting the onnx model: {e}")


if __name__ == '__main__':

    # train model
    model = train_model()
