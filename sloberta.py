from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

import numpy as np
from tqdm import tqdm
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
model = AutoModel.from_pretrained("EMBEDDIA/sloberta")
model.eval()  # inference mode


def get_sloberta_embedding(text, pooling="cls"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    if pooling == "cls":
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    elif pooling == "mean":
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def build_text(article):
    title = article.get("title", "") or ""
    lead = article.get("lead", "") or ""
    paragraphs = article.get("paragraphs", []) or []

    # Filter out empty paragraphs and join with space
    paragraphs_text = " ".join(p for p in paragraphs if p)

    # Join all parts with a space in between
    full_text = ". ".join(part for part in [title, lead, paragraphs_text] if part.strip())

    return full_text

def generate_and_save_embeddings(json_path, output_path, pooling="cls"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    partial_file = "sloberta_features_partial.npy"
    features = []
    start_idx = 0

    # Check if partial file exists and load it
    if os.path.exists(partial_file):
        print(f"Found partial file: {partial_file}")
        features = list(np.load(partial_file))
        start_idx = len(features)
        print(f"Resuming from index {start_idx} of {len(data)}")

    for i in tqdm(range(start_idx, len(data)), desc="Generating embeddings"):
        article = data[i]
        text = build_text(article)

        if not text.strip():
            emb = np.zeros(768)  # fallback for empty text
        else:
            emb = get_sloberta_embedding(text, pooling=pooling)

        features.append(emb)

        #print(f"TEXT:\n{text}")
        #print(f"EMBEDDING:\n{emb}")

        #break
        # Autosave every 100
        
        if (i + 1) % 1000 == 0 or (i + 1) == len(data):
            np.save(partial_file, np.stack(features))
            print(f"Saved {i + 1} embeddings to {partial_file}")
        i += 1

    features = np.stack(features)  # shape: (N, 768)
    np.save(output_path, features)
    print(f"Saved {features.shape[0]} embeddings with {features.shape[1]} features to {output_path}")
    os.remove(partial_file)  # clean up



class Model_Neural_Network(nn.Module):

    def __init__(self, input_dim, lr=0.01, epochs=500, lambda_reg=0.0001, weight_dec=0.00):
        super().__init__()

        # create a model
        self.input_size = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            #nn.Linear(256, 64),
            #nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(256, 1),  # Output = 1 value for regression
            nn.ReLU()           # to ensure values are >= 0
        )

        # store other variables
        self.lr = lr 
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.wd = weight_dec


    def forward(self, x):
        return self.model(x)


    def soft_threshold(self, param, lmbd):
        with torch.no_grad():
            # postavi na 0 utezi, ki so manjse od lmbd
            # tiste, ko so vecje, pa zmanjsa za lmbd
            # param.copy_() je funkcija, ki lahko neposredno popravi utezi
            param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))



    def fit(self, X_train, y_train):
        print("************\nTRAINING")

        # 2. Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        #y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
        y_tensor = torch.tensor(np.log1p(y_train), dtype=torch.float32).view(-1, 1)


        # 3. Define optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.MSELoss()


        # 4. Training loop
        for epoch in range(self.epochs):
            self.train()

            pred = self.model(X_tensor)

            mse_loss = loss_fn(pred, y_tensor)

            # add L1 regularization
            l1_norm = sum(param.abs().sum() for name, param in self.named_parameters() if 'weight' in name)
            loss = mse_loss + self.lambda_reg * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # apply soft thresholding
            for name, param in self.named_parameters():
                if 'weight' in name:
                    self.soft_threshold(param, self.lambda_reg)

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")


    def predict(self, X, attributes_list):
 
        self.eval()
        with torch.no_grad():
            
            #preds_log = self(X_tensor)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            #preds = self(X_tensor)

            # Apply inverse log1p to get real prediction values
            preds_log = self(X_tensor)
            preds = torch.expm1(preds_log).squeeze().numpy()

            
            # izpiši pomembnosti značilk

            # Find first linear layer and get its weights
            first_linear = None
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break

            if first_linear is None:
                print("No linear layer found in model.")
                return preds

            weights = first_linear.weight.data.cpu().numpy()

            if weights.ndim > 1:
                weights = weights.mean(axis=0)


            # Count zero weights (due to L1 soft-thresholding)
            zero_weights = np.sum(weights == 0)
            total_weights = len(weights)
            print(f"\nFeature sparsity: {zero_weights}/{total_weights} weights are zero")

            features = attributes_list
            sorted_weights = sorted(
                zip(weights, features), 
                key=lambda x: abs(x[0]), 
                reverse=True
            )
        
            print("Feature importances:")
            idx = 0
            for weight, name in sorted_weights:
                print(f"{name:9s}: {weight:7.4f}")
                idx += 1
                if idx > 10:
                    break         

            return preds

    def get_scaler(self):
        return self.scaler




def load_X_y(X_path, y_path):
    
    # load X
    X = np.load(X_path)
    print(f"X shape: {X.shape}")

    # load dataset
    with open(y_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # extract comments
    number_of_comments = []     # done
    for article in tqdm(data, desc="data extraction"):
        # target value: number of comments
        n_comments = article["n_comments"]
        number_of_comments.append(n_comments)

    print(f"y length: {len(number_of_comments)}")

    return X, number_of_comments


if __name__ == "__main__":
    """
    #data_path = "../data/rtvslo_train.json"
    #embeddings_path = "sloberta_features_mean.npy"
    #generate_and_save_embeddings(data_path, embeddings_path, pooling="mean")

    data_path = "../data/rtvslo_test.json"
    embeddings_path = "TEST_sloberta_features_mean.npy"
    generate_and_save_embeddings(data_path, embeddings_path, pooling="mean")
  
    """
    # load X and y set
    X_path = "sloberta_features.npy"
    y_path = "../data/rtvslo_train.json"
    X, y = load_X_y(X_path, y_path)
    attributes_len = X.shape[1]
    print(attributes_len)

    # create training split
    # using random because we're training purely on text
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # send it to nn
    model = Model_Neural_Network(input_dim=attributes_len)
    model.fit(X_train_scaled, y_train)

    # Evaluate on test split
    attributes_list = [f"sloberta_{i}" for i in range(attributes_len)]
    y_pred = model.predict(X_val_scaled, attributes_list)

    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).view(-1,1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    final_mse = nn.MSELoss()(y_pred_tensor, y_val_tensor).item()
    print(f"\nMSE on local test data: {final_mse:.4f}")
    
