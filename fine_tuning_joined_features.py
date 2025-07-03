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


class Model_Neural_Network(nn.Module):

    def __init__(self, input_dim, lr=0.01, epochs=40, lambda_reg=0.00, weight_dec=0.001):
        super().__init__()

        # create a model
        self.input_size = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 832),
            nn.BatchNorm1d(832),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(832, 512),
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
            nn.Dropout(0.1),

            nn.Linear(64, 1),  # Output = 1 value for regression
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



    def fit(self, X_train, y_train, batch_size=64):
        print("************\nTRAINING")

        # 2. Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        #y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
        y_tensor = torch.tensor(np.log1p(y_train), dtype=torch.float32).view(-1, 1)


        # batch learning
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. Define optimizer and loss
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.L1Loss()


        # 4. Training loop
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0

            for batch_X, batch_y in dataloader:

                pred = self.model(batch_X)

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


    def predict(self, X, attributes_list):
 
        self.eval()
        with torch.no_grad():
            
            #preds_log = self(X_tensor)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            #preds = self(X_tensor)

            # Apply inverse log1p to get real prediction values
            preds_log = self(X_tensor)
            preds = torch.expm1(preds_log).squeeze().numpy()

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

    return meta_attributes + bert_attributes


def cross_validate_model(X, y, batch_size, lr, lambda_reg, weight_dec, epoch, k=5):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        #print(f"\nFold {fold+1}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # get attributes 
        attributes_list = all_attributes()
        attributes_len = len(attributes_list)
        #print(f"Attributes list has now {attributes_len} attributes")

        # train the model
        model = Model_Neural_Network(
            input_dim=attributes_len,
            lr=lr,
            epochs=epoch,
            lambda_reg=lambda_reg,
            weight_dec=weight_dec
            )
        model.fit(X_train_scaled, y_train, batch_size=batch_size)

        # evaluate the model
        y_pred = model.predict(X_val_scaled, attributes_list)
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)
        #print(f"Fold MAE: {mae:.2f}")
    return np.mean(mae_scores)


#BATCH_SIZE = 64

#BATCH_SIZES = [16, 32, 64, 128]
#LEARNING_RATES = [0.001, 0.005, 0.01, 0.05, 0.1]
#LAMBDA_REGRESSIONS = [0.0]
#WEIGHT_DECAYS = [0.0001, 0.001]

EPOCHS = [30, 40, 50]
BATCH_SIZES = [16, 32]
LEARNING_RATES = [0.01]
LAMBDA_REGRESSIONS = [0.0]
WEIGHT_DECAYS = [0.0001]

if __name__ == "__main__":

    # load X 
    X_bert = np.load("sloberta_features_mean.npy")            # Shape: (N, 768)
    X_meta = np.load("meta_features.npy")                     # Shape: (N, 64)
    #print(f"bert features shape: {X_bert.shape}\nmeta features shape: {X_meta.shape}")

    X = np.concatenate([X_bert, X_meta], axis=1)     # Shape: (N, 832)
    #print(f"X shape: {X.shape}")

    # load y 
    y_path = "../data/rtvslo_train.json"
    y_full = load_json_data(y_path)
    y_comments = [article["n_comments"] for article in y_full]
    y = np.array(y_comments, dtype=np.float32)
    #print(f"y shape: {y.shape}")

    # train test split
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    """
    # fine tuning
    model_idx = 1
    best_mae = float('inf')
    best_params = None

    for batch_size in BATCH_SIZES:
        for lr in LEARNING_RATES:
            for lambda_reg in LAMBDA_REGRESSIONS:
                for weight_dec in WEIGHT_DECAYS:
                    if batch_size == 16 and lr == 0.001 and lambda_reg == 0.0:
                        continue

                    average_mae = cross_validate_model(
                        X, y,
                        batch_size=batch_size,
                        lr=lr,
                        lambda_reg=lambda_reg,
                        weight_dec=weight_dec
                    )

                    if average_mae < best_mae:
                        best_mae = average_mae
                        best_params = (batch_size, lr, lambda_reg, weight_dec)

                    print(f"******MODEL NUMBER {model_idx}******")
                    print(f"batch size={batch_size}, lr={lr}, lam reg={lambda_reg}, weight decay={weight_dec}")
                    print(f"-> MAE = {average_mae}") 
                    model_idx += 1

    print("\nBest configuration:")
    print(f"Batch size: {best_params[0]}, LR: {best_params[1]}, L1: {best_params[2]}, WD: {best_params[3]}")
    print(f"Best MAE: {best_mae:.4f}")
    """
    # fine tuning
    model_idx = 1
    best_mae = float('inf')
    best_params = None

    for epoch in EPOCHS:
        average_mae = cross_validate_model(
            X, y,
            batch_size=16,
            lr=0.01,
            lambda_reg=0.0,
            weight_dec=0.0001,
            epoch=epoch
        )

        if average_mae < best_mae:
            best_mae = average_mae
            #best_params = (batch_size, lr, lambda_reg, weight_dec, epoch)
            best_params = epoch

        print(f"******MODEL NUMBER {model_idx}******")
        #print(f"batch size={batch_size}, lr={lr}, lam reg={lambda_reg}, weight decay={weight_dec}")
        print(f"epochs = {epoch}")
        print(f"-> MAE = {average_mae}") 
        model_idx += 1

    print("\nBest configuration:")
    #print(f"Batch size: {best_params[0]}, LR: {best_params[1]}, L1: {best_params[2]}, WD: {best_params[3]}")
    print(f"epochs: {best_params}")
    print(f"Best MAE: {best_mae:.4f}")




    #print(f"\nAverage MAE across folds: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")

    # TODO: here we're supposed to build a final model with fine tuned parameters on all data



    """
    # FINAL OUTPUT

    # load X 
    X_test_bert = np.load("TEST_sloberta_features_mean.npy")
    X_test_meta = np.load("TEST_meta_features.npy")                     # Shape: (N, 64)
    print(f"bert features shape: {X_test_bert.shape}\nmeta features shape: {X_test_meta.shape}")

    X_final_test = np.concatenate([X_test_bert, X_test_meta], axis=1)     # Shape: (N, 832)
    print(f"X shape: {X_final_test.shape}")

    attributes_final_list = all_attributes()
    attributes_len = len(attributes_final_list)

    X_final_scaled = scaler.transform(X_final_test)
    y_pred_final = model.predict(X_final_scaled, attributes_final_list)

    # in example.txt we get predicted number of comments for each article in test set
    np.savetxt('TEST_RESULTS_joined_features.txt', y_pred_final, fmt='%f')
    """
