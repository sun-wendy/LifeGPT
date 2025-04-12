import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import typing
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import foundation_models
import metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Tokenizer:
    def __init__(self, n_pad: int, device: torch.device, pad_byte: int = 0):
        self.n_pad = n_pad
        self.device = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad, f"n_pad is too small, use {len(base)} or greater."
        tensor = torch.Tensor(base)
        return tensor.long().to(self.device)

    def texts_to_sequences(self, texts: typing.List[str], encoding="utf8", do_padding=True):
        sentences = [self.tokenize_str(sentence, do_padding=do_padding).unsqueeze(0) for sentence in texts]
        return torch.cat(sentences, dim=0).to(self.device)

    def sequences_to_texts(self, texts: torch.Tensor, encoding="utf8"):
        out = []
        for seq in texts:
            chars = []
            i = 0
            while i < len(seq) and seq[i] != 0:
                chars.append(int(seq[i]))
                i += 1
            try:
                out.append(bytes(chars).decode(encoding))
            except:
                pass
        return out

def empty_cuda_cache():
    torch.cuda.empty_cache()

def extract_sample(string_input, start_token='[', end_token=']'):
    i = string_input.find(start_token)
    j = string_input.find(end_token)
    return string_input[i+1:j]

def extract_task(string_input, end_task_token='>'):
    j = string_input.find(end_task_token)
    return string_input[:j+1]

def load_model(model_path, max_length, num_words):
    empty_cuda_cache()
    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=256,
            depth=12,
            heads=8,
            attn_dim_head=64,
            rotary_pos_emb=True,
            attn_flash=True
        )
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"Model loaded from {model_path}")
    return model


def cell_to_image(cell_str, grid_size=(32, 32)):
    """
    Converts a string representing a flattened binary grid into a 3-channel image.
    Expects that cell_str contains exactly grid_size[0]*grid_size[1] bits (0/1).
    """
    digits = re.findall(r'[01]', cell_str)
    arr = np.array(list(map(int, digits)))
    if len(arr) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0] * grid_size[1]} digits, but got {len(arr)}.")
    arr = arr.reshape(grid_size).astype(np.float32)
    # Normalize to [0, 1] if not already
    img = np.stack([arr, arr, arr], axis=-1)  # (H, W, 3)
    return img


class MacrosDataset(Dataset):
    def __init__(self, csv_file, tokenizer, model, clip, state_columns, grid_size=(32,32)):
        """
        csv_file: path to the CSV with Conway states.
        tokenizer: instance to tokenize text prompts for the LifeGPT model.
        model: pretrained LifeGPT model to compute latent representations.
        clip: an instance of the CLIP helper for image embeddings.
        state_columns: list of column names containing the states (e.g., ["State 1", ..., "State 10"])
        grid_size: expected dimensions of the state grid.
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.model = model
        self.clip = clip
        self.state_columns = state_columns
        self.grid_size = grid_size
        self.device = next(model.parameters()).device  # assume model is on proper device

    def __len__(self):
        return len(self.df)

    def encode_initial_state(self, state_str):
        """
        Constructs a text prompt with the initial state and returns its latent representation.
        Example prompt: "@PredictNextState<STATE1>"
        """
        prompt = f"@PredictNextState<{state_str}>"
        tokens = self.tokenizer.texts_to_sequences([prompt], do_padding=False)
        tokens = tokens.to(self.device).long()
        with torch.no_grad():
            outputs, hidden_states = self.model.net(tokens, return_intermediates=True)
            hidden = hidden_states.hiddens[-1]
        fixed_prefix_length = len("@PredictNextState<")  # Number of characters/tokens in the fixed part
        state_token_start = fixed_prefix_length
        state_token_end = state_token_start + len(state_str)
        latent_tokens = hidden[0, state_token_start:state_token_end, :]  # shape: (L, D)
        latent = latent_tokens.mean(dim=0)  # shape: (D,)
        return latent

    def compute_open_endedness(self, row):
        """
        Converts each state in the row to an image, embeds it using CLIP,
        and computes the open-endedness score from the sequence of embeddings.
        """
        images = []
        for col in self.state_columns:
            state_str = str(row[col])
            try:
                img = cell_to_image(state_str, grid_size=self.grid_size)
                images.append(img)
            except Exception as e:
                print(f"Error processing state {col}: {e}")
        # Embed each image using CLIP
        embeddings = []
        for img in images:
            emb = self.clip.embed_img(img)
            embeddings.append(emb)
        embeddings_tensor = torch.stack(embeddings, dim=0)  # (T, D_clip)
        print(embeddings_tensor.shape, flush=True)
        score = metrics.calc_open_endedness_score(embeddings_tensor)
        print(score, flush=True)
        return score
    
    def compute_supervised_target(self, row):
        """
        Computes the supervised target score using final state image and a text prompt ("glider")
        via calc_supervised_target_score. This function assumes a single final state image (T=1)
        and a single text embedding (T2=1), so the kernel becomes a 1x1 matrix.
        """
        final_state_str = str(row[self.state_columns[-1]])
        try:
            final_img = cell_to_image(final_state_str, grid_size=self.grid_size)
        except Exception as e:
            print(f"Error processing final state: {e}")
            return 0.0
        
        final_img_emb = self.clip.embed_img(final_img).unsqueeze(0)  # shape: (1, D)
        text_emb = self.clip.embed_txt(["glider"])  # shape: (1, D)
        target = metrics.calc_supervised_target_score(final_img_emb, text_emb)
        return target 

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        initial_state = row[self.state_columns[0]]
        latent = self.encode_initial_state(initial_state)  # shape: (D,)
        # score = self.compute_open_endedness(row)  # scalar open-endedness score
        score = self.compute_supervised_target(row)
        print(score, flush=True)
        return latent, torch.tensor(score, dtype=torch.float32)


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # maps latent vector to a single scalar

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    epoch=50
    num_words=256
    max_length = 2071 #len("@PredictNextState<1024-bits> [1024-bits]$")
    model_path = f"LifeGPT/model_parameters/07_22_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_On_Broad_Entropy_Homog_2025-04-12 06-16-41/LifeGPT_v7_epoch_2.pt"
    model = load_model(model_path, max_length, num_words)
    model.eval()

    tokenizer = Tokenizer(n_pad=max_length, device=device)

    clip = foundation_models.create_foundation_model("clip")

    train_file = "Conway_GPT/EXAMPLE_conway_states_0_1_100by32by32by256_toroidal_20250412_075417.csv"
    state_columns = [f"State {i}" for i in range(1, 33)]
    dataset = MacrosDataset(train_file, tokenizer, model, clip, state_columns, grid_size=(32,32))
    dataset = Subset(dataset, indices=range(100))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataloader))

    linear_probe = LinearProbe(input_dim=256)
    linear_probe = linear_probe.to(device)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        linear_probe.train()
        epoch_loss = 0.0
        for batch in dataloader:
            latents, scores = batch  # latents: (B, D), scores: (B,)
            print(latents.shape)
            latents = latents.to(device)
            scores = scores.to(device)
            optimizer.zero_grad()
            preds = linear_probe(latents).squeeze(-1)  # (B,)
            loss = loss_fn(preds, scores)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * latents.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    """
    # Load in Data from CSVs
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    # Define training size and special characters
    TRAIN_SIZE = 10000
    TEST_SIZE = 1000
    start_char = '@'
    end_char = '$'
    mask_char = ['_']

    # Function to generate data
    def generate_data(df, future_steps):
        X_data = []
        for i in range(len(df['State 1'])):
            future_state_col = f'State {future_steps}'
            if future_state_col in df.columns:
                str_ = f"{start_char}PredictNextState<{df['State 1'][i]}> [{df[future_state_col][i]}]{end_char}"
                X_data.append(str_)
        return X_data

    # Generate datasets for different future steps
    future_steps_list = [2, 3, 5, 10]
    X_data_train = {steps: generate_data(df_train, steps) for steps in future_steps_list}
    X_data_val = {steps: generate_data(df_val, steps) for steps in future_steps_list}
    X_data_test = {steps: generate_data(df_test, steps) for steps in future_steps_list}
    print(X_data_test[2][0])
    print(X_data_test[2][0][0])

    # Print the number of sequences in each dataset
    for steps in future_steps_list:
        print(f"Train set for {steps} future steps: {len(X_data_train[steps])} sequences")
        print(f"Validation set for {steps} future steps: {len(X_data_val[steps])} sequences")
    """
