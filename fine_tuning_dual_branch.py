import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

import numpy as np
from tqdm import tqdm
import json
import os
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


class Model_Dual_Branch(nn.Module):

    def __init__(self, input_dim_bert=768, input_dim_meta=64, lr=0.01, epochs=40, lambda_reg=0.00, weight_dec=0.0001):
        super().__init__()

        # model 1 : bert model
        self.model_bert = nn.Sequential(
            nn.Linear(input_dim_bert, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # model 2 : metadata model
        self.model_meta = nn.Sequential(
            nn.Linear(input_dim_meta, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(0.2),
        )

        # final layer that connects these two models
        self.model_final = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
            nn.ReLU(),
            #nn.Dropout(0.2),
        )

        # store other variables
        self.lr = lr 
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.wd = weight_dec


    def forward(self, x_bert, x_meta):

        bert_out = self.model_bert(x_bert)
        meta_out = self.model_meta(x_meta)
        combined = torch.cat((bert_out, meta_out), dim=1)

        return self.model_final(combined)


    def soft_threshold(self, param, lmbd):
        with torch.no_grad():
            # postavi na 0 utezi, ki so manjse od lmbd
            # tiste, ko so vecje, pa zmanjsa za lmbd
            # param.copy_() je funkcija, ki lahko neposredno popravi utezi
            param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))



    def fit(self, X_bert_tensor, X_meta_tensor, y_tensor, batch_size=64):
        print("************\nTRAINING")

        # batch learning
        dataset = TensorDataset(X_bert_tensor, X_meta_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. Define optimizer and loss
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.L1Loss()


        # 4. Training loop
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0

            for batch_bert, batch_meta, batch_y in dataloader:

                pred = self.forward(batch_bert, batch_meta)


                mse_loss = loss_fn(pred, batch_y)

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
                
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch} - Avg Loss: {total_loss / len(dataloader):.4f}")


    def predict(self, X_bert_tensor, X_meta_tensor, attributes_list):
 
        self.eval()
        with torch.no_grad():
            
            #preds_log = self(X_tensor)
            #preds = self(X_tensor)

            # Apply inverse log1p to get real prediction values
            #preds_log = self(X_tensor)
            #preds = torch.expm1(preds_log).squeeze().numpy()

            #preds = self.forward(X_bert_tensor, X_meta_tensor).squeeze().numpy()
            preds_log = self.forward(X_bert_tensor, X_meta_tensor)
            preds = np.expm1(preds_log).squeeze().numpy()

            """
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
                if idx > 20:
                    break      
            """

            return preds



def load_json_data(path):
    """Function for opening json data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def all_attributes():

    meta_attributes = ['paragraph_length', 'group_crna-kronika', 'group_gospodarstvo', 'group_kolumne', 'group_kultura', 
        'group_okolje', 'group_slovenija', 'group_sport', 'group_stevilke', 'group_svet', 'group_zabava-in-slog', 
        'group_znanost-in-tehnologija', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 
        'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 
        'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_1', 'month_2', 'month_3', 'month_4', 
        'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'year_2017', 'year_2018', 
        'year_2019', 'year_2020', 'year_2021', 'year_2022', 'year_2023', 'year_2024', 'year_2025', 'day_of_week_0', 
        'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6']

    bert_attributes = [f"sloberta_{i}" for i in range(768)]

    return meta_attributes, bert_attributes


def cross_validate_dual_branch(X_bert, X_meta, y, batch_size, lr, lambda_reg, weight_dec, epoch, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, val_idx in kf.split(X_bert):
        X_train_b, X_val_b = X_bert[train_idx], X_bert[val_idx]
        X_train_m, X_val_m = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # scale each branch separately
        scaler_b = StandardScaler().fit(X_train_b)
        scaler_m = StandardScaler().fit(X_train_m)

        X_train_b_scaled = scaler_b.transform(X_train_b)
        X_val_b_scaled = scaler_b.transform(X_val_b)
        X_train_m_scaled = scaler_m.transform(X_train_m)
        X_val_m_scaled = scaler_m.transform(X_val_m)

        # tensors
        X_train_b_tensor = torch.tensor(X_train_b_scaled, dtype=torch.float32)
        X_val_b_tensor = torch.tensor(X_val_b_scaled, dtype=torch.float32)
        X_train_m_tensor = torch.tensor(X_train_m_scaled, dtype=torch.float32)
        X_val_m_tensor = torch.tensor(X_val_m_scaled, dtype=torch.float32)
        #y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_train_tensor = torch.tensor(np.log1p(y_train), dtype=torch.float32).view(-1, 1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        attributes_meta, attributes_bert = all_attributes()
        attributes_list = attributes_meta + attributes_bert

        model = Model_Dual_Branch(
            input_dim_bert=len(attributes_bert),
            input_dim_meta=len(attributes_meta),
            lr=lr,
            epochs=epoch,
            lambda_reg=lambda_reg,
            weight_dec=weight_dec
        )

        model.fit(X_train_b_tensor, X_train_m_tensor, y_train_tensor, batch_size=batch_size)
        y_pred = model.predict(X_val_b_tensor, X_val_m_tensor, attributes_list)
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)

    return np.mean(mae_scores)


if __name__ == "__main__":

    # load X 
    X_bert = np.load("sloberta_features_mean.npy")            # Shape: (N, 768)
    X_meta = np.load("meta_features.npy")                     # Shape: (N, 64)
    #print(f"bert features shape: {X_bert.shape}\nmeta features shape: {X_meta.shape}")

    #X = np.concatenate([X_bert, X_meta], axis=1)     # Shape: (N, 832)
    #print(f"X shape: {X.shape}")

    # load y 
    y_path = "../data/rtvslo_train.json"
    y_full = load_json_data(y_path)
    y_comments = [article["n_comments"] for article in y_full]
    y = np.array(y_comments, dtype=np.float32)
    #print(f"y shape: {y.shape}")


    # hyperparams
    EPOCHS = [60, 70, 80]
    BATCH_SIZES = [16]
    LEARNING_RATES = [0.01]
    LAMBDA_REGRESSIONS = [0.0]
    WEIGHT_DECAYS = [0.0001]

    model_idx = 1
    best_mae = float("inf")
    best_config = None

    for epoch in EPOCHS:
        for batch_size in BATCH_SIZES:
            for lr in LEARNING_RATES:
                for lambda_reg in LAMBDA_REGRESSIONS:
                    for weight_dec in WEIGHT_DECAYS:

                        mae = cross_validate_dual_branch(
                            X_bert, X_meta, y,
                            batch_size=batch_size,
                            lr=lr,
                            lambda_reg=lambda_reg,
                            weight_dec=weight_dec,
                            epoch=epoch
                        )

                        print(f"\n****** MODEL {model_idx} ******")
                        print(f"batch={batch_size}, lr={lr}, l1={lambda_reg}, wd={weight_dec}, epochs={epoch}")
                        print(f"-> MAE: {mae:.4f}")

                        if mae < best_mae:
                            best_mae = mae
                            best_config = (batch_size, lr, lambda_reg, weight_dec, epoch)

                        model_idx += 1

    print("\nBest configuration:")
    print(f"batch size: {best_config[0]}, lr: {best_config[1]}, l1: {best_config[2]}, wd: {best_config[3]}, epochs: {best_config[4]}")
    print(f"Best MAE: {best_mae:.4f}")

