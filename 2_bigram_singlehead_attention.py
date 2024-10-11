import torch 
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)  # for reproducibility

SEP = 50 * '-'

# hyperparameters ----------------------------------------------------------------------------------
batch_size = 32  # how many independent sequences will we process in parallel
block_size = 8  # what i sthe maximum context length for predictions
max_iters = 5000  # how many iterations to train for
eval_interval = 500  # how often to evaluate the model
learning_rate = 1e-3  # how fast we update the weights, self-attention cannot tolerate high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # check if GPU is available
eval_iters = 200  # how many batches to average for evaluation
n_embd = 32  # number of embedding dimensions

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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# simple bigram model ------------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits from the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # positional embeddings
        self.sa_head = Head(n_embd)  # self-attention head
        self.lm_head = nn.Linear(n_embd, vocab_size)  # output layer

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) = Batch, Time (block_size), Channels (vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)  # apply one head of self-attention. (B, T, C)
        logits = self.lm_head(x)  # decoder head (B, T, vocab_size)

        if targets is None:
            loss = None

        else:
            # reshape the logits to be (B*T, C) and the targets to be (B*T) so we can compute the loss
            B, T, C = logits.shape  # unpack batch, time, channels
            logits = logits.view(B*T, C)  # flatten the Time and Batch dimensions
            targets = targets.view(B*T)

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


model = BigramLanguageModel()
m = model.to(device)

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
    logits, loss = m(xb, yb)  # calling the model and passing in the input and the targets
    optimizer.zero_grad(set_to_none=True)  # clear previous gradients
    loss.backward()  # compute new gradients
    optimizer.step()  # update the weights

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # initialize context to be a single token
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))  # generate 100 new tokens
