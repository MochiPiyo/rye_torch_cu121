import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torchtext.datasets import WMT14
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import Field, BucketIterator
from torch.nn.utils.rnn import pad_sequence
import spacy
print(torch.__version__)


# これを作成
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    # x: Batch * vocab * dim
    def forward(self, x):
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        # 最大長までの
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            # 0..self.embed_dim by stride 2
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 2] = math.cos(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
        
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)
    
    # batch * sequence_len * dim
    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # batch * seq_len * n_heads * single_head_dim
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        # (batch_size, n_heads, seq_len, single_head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)

        product = torch.matmul(q, k_adjusted)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        # dividing by square roo tof key dimension
        product = product / math.sqrt(self.single_head_dim)

        # applying softmax
        scores = F.softmax(product, dim=-1)
        # multiply with value matrix
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)

        # Linear
        output = self.out(concat)
        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        # Multi Head Attention and Add&Norm
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + value

        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out

        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        return norm2_out
    

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_layer = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_layer(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
    
    def forward(self, key, query, x, mask):
        # we need to pass mask only to first attention
        attention = self.attention(x, x, x, mask=mask)
        value = self.dropout(self.norm(attention + x))

        # key and query are come from encoder
        out = self.transformer_block(key, query, value)
        return out
    

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()

        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) for _ in range(num_layers)]
        )

        # 最終の単語解釈Linear
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.droptout = nn.Dropout(0.2)
    
    def forward(self, x, encoder_out, mask):
        x = self.word_embedding(x)
        x = self.position_embedding(x)
        x = self.droptout(x)

        for layer in self.layers:
            x = layer(encoder_out, x , encoder_out, mask)

        out = F.softmax(self.fc_out(x))
        return out


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4, n_heads=8):
        super(Transformer, self).__init__()

        self.target_vocab_size = target_vocab_size
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers, expansion_factor, n_heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers, expansion_factor, n_heads)

    def make_trg_mask(self, trg):
        # trg: target sequence
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def decode(self, src, trg):
        # src: input to encoder
        # trg: input to decoder
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        _batch_size, seq_len = src.shape[0], src.shape[1]

        # outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len):
            out = self.decoder(out, enc_out, trg_mask)
            # taking the last token
            out = out[:, -1, :]
            
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)

        return out_labels

    def forward(self, src, trg):
        # src: input to encoder
        # trg: input to decoder

        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs




# Load spacy tokenizers
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Define fields
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Load dataset
train_data, valid_data, test_data = WMT14.splits(exts=('.de', '.en'), fields=(SRC, TRG))

# Build vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define hyperparameters
BATCH_SIZE = 128
MAX_LEN = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)


# Instantiate the model
model = Transformer(
    embed_dim=512,
    src_vocab_size=len(SRC.vocab),
    target_vocab_size=len(TRG.vocab),
    seq_length=MAX_LEN,
    num_layers=6,
    expansion_factor=4,
    n_heads=8
).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Define training and evaluation functions
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Training loop
N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-wmt14-en-de.pt')
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Evaluate on test set
model.load_state_dict(torch.load('transformer-wmt14-en-de.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}')