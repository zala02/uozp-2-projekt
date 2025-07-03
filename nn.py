# basic libraries
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime


# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset


def load_json_data(path):
    """Function for opening json data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



class PrepareData():
    """Class for reading a json file with articles, extracting atributes, splitting data into train test set."""

    def __init__(self, filename, batch_size=64):

        print("***\nInitialization of PrepareData started")

        # store the variables
        self.filename = filename
        
        # store the original data since im going to modify it (just in case, can delete that later)
        self.og_data = load_json_data(self.filename)

        # other variables
        self.attributes = []
        self.target_list = []
        self.scaler = StandardScaler()
        
        # first we load data, possibly process it, add attributes if needed
        self.data = self._load_and_process()

        # divide into train validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = self._create_train_test()

        print("***\nSuccessfully initialized class PrepareData")


    def _load_and_process(self):
        """Load data, drop nan values, add atributes."""

        print("***\nLoading and processing data started")

        data = load_json_data(self.filename)

        # extract attributes and y
        # attributes: group (from url), hour and day in week when published, length of paragraph
        # y: number of comments 
        groups = []                 # done
        hours = []                  # done
        months = []                 # done
        years = []                  # done
        days_of_week = []           # done
        paragraph_length = []       # done
        authors = []                # done
        number_of_comments = []     # done
        for article in tqdm(data, desc="data extraction"):

            # attribute 1 : paragraph length
            p_length = 0
            if article["paragraphs"] != []:
                for paragraph in article["paragraphs"]:
                    p_length += len(paragraph)

            elif article["lead"] != "":
                p_length = len(article["lead"])

            paragraph_length.append(p_length)

            # attribute 2 : group from url 
            url = article["url"]
            url_split = url.split("/")
            group = url_split[3]
            # 'https:', '', 'www.rtvslo.si' so vedno prvi trije, vzamem cetrtega torej
            groups.append(group)

            # attribute 3 : hour
            # attribute 4 : month
            # attribute 5 : year
            # attribute 6 : day of week
            date_string = article["date"]
            dt = datetime.fromisoformat(date_string)

            hour = dt.hour
            month = dt.month
            year = dt.year
            day_of_week = dt.weekday()

            hours.append(hour)
            months.append(month)
            years.append(year)
            days_of_week.append(day_of_week)

            # attribute 7 : author
            if "authors" not in article or article["authors"] == "":
                author = "Unknown"
            else:
                author = article["authors"]
            authors.append(author)

            # target value: number of comments
            n_comments = article["n_comments"]
            number_of_comments.append(n_comments)
        """
        print(f"Groups: {len(groups)}")
        print(f"Hours: {len(hours)}")
        print(f"Months: {len(months)}")
        print(f"Years: {len(years)}")
        print(f"Days of week: {len(days_of_week)}")
        print(f"Paragraph lengths: {len(paragraph_length)}")
        print(f"Authors: {len(authors)}")
        print(f"Number of comments: {len(number_of_comments)}")

        print(f"Unique groups: {len(set(groups))}")
        print(f"Unique hours: {len(set(hours))}")
        print(f"Unique months: {len(set(months))}")
        print(f"Unique years: {len(set(years))}")
        print(f"Unique days of week: {len(set(days_of_week))}")
        print(f"Unique paragraph lengths: {len(set(paragraph_length))}")
        #print(f"Unique authors: {len(set(authors))}")
        print(f"Unique number of comments: {len(set(number_of_comments))}")
        """

        # transform into panda format
        data_panda = pd.DataFrame({
            "group": groups,
            "hour": hours,
            "month": months,
            "year": years,
            "day_of_week": days_of_week,
            "paragraph_length": paragraph_length,
            "number_of_comments": number_of_comments  # target
        })

        # maybe drop nan but idk if theres any
        #print(f"Nan values:\n{data_panda.isna().any()} ")

        # one-hot encode group, hour, month, year, day_of_week
        data_encoded = pd.get_dummies(
            data_panda,
            columns=["group", "hour", "month", "year", "day_of_week"],
            drop_first=False  # set to True if you want to avoid multicollinearity
        )
        #print(data_encoded.head())
        #print(f"Final shape: {data_encoded.shape}")

        self.attributes = data_encoded.drop("number_of_comments", axis=1).columns.tolist()
        self.target_list = ["number_of_comments"]


        print(f"For now we have following {len(self.attributes)} attributes: {self.attributes}")
        #print(f"target list: {self.target_list}")
        
        print(f"Loading and processing data finished.")

        #X_meta = data_encoded.drop("number_of_comments", axis=1).values.astype(np.float32)
        #np.save("meta_features.npy", X_meta)
        #print(f"Saved metadata features: {X_meta.shape}")


        return data_encoded


    def _create_train_test(self):
        """Divide data to <train_size> and <predict_size>, drop <gap_size>."""

        print("***\nCreating train and test sets started")

        X = self.data[self.attributes].values.astype(np.float32)   #np.int_
        y = self.data[self.target_list].values.astype(np.float32)
        #print(X)
        #print(y)

        """
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


        print("Creating train and test sets finished")

        return X_train, X_val, y_train, y_val


    def get_train_data(self):
        """Creates and returns a PyTorch data object for training."""

        #X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        #y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).view(-1,1)
        #return X_train_tensor, y_train_tensor
        return self.X_train, self.y_train


    def get_test_data(self):
        """Returns validation data as PyTorch tensors."""

        #X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32).view(-1,1)
        #return X_val_tensor, y_val_tensor
        return self.X_val, y_val_tensor

    def get_attributes(self):
        """Returns a list of all attributes."""
        return self.attributes


class Model_Neural_Network(nn.Module):

    def __init__(self, input_dim, lr=0.001, epochs=10, lambda_reg=0.00, weight_dec=0.001):
        super().__init__()

        # create a model
        self.input_size = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1),  # Output = 1 value for regression
            nn.ReLU()
        )

        # store other variables
        self.lr = lr 
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.wd = weight_dec

        # store scaled for standardizing data
        self.scaler = StandardScaler()


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

        # 1. Fit the scaler on all training data
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)


        # 2. Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
        #y_tensor = torch.tensor(np.log1p(y_train), dtype=torch.float32).view(-1, 1)

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
            
            # standardize data
            #X_np = X.numpy()
            X_normed = self.scaler.transform(X)
            X_tensor = torch.tensor(X_normed, dtype=torch.float32)
            
            #preds_log = self(X_tensor)

            # Apply inverse log1p to get real prediction values
            #preds = torch.expm1(preds_log).squeeze().numpy()
            preds = self(X_tensor)
            
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

            features = attributes_list
            sorted_weights = sorted(zip(weights, features), key=lambda x: abs(x[0]), reverse=True)
        
            print("Feature importances:")
            idx = 0
            for weight, name in sorted_weights:
                print(f"{name:9s}: {weight:7.4f}")
                idx += 1
                if idx > 10:
                    break         
            """
            return preds

    def get_scaler(self):
        return self.scaler

def prepare_test_data(filename, attributes_list):

    print("***\nLoading and processing test data started")

    data = load_json_data(filename)

    # extract attributes and y
    # attributes: group (from url), hour and day in week when published, length of paragraph
    # y: number of comments 
    groups = []                 # done
    hours = []                  # done
    months = []                 # done
    years = []                  # done
    days_of_week = []           # done
    paragraph_length = []       # done
    authors = []                # done

    for article in tqdm(data, desc="data extraction"):

        # attribute 1 : paragraph length
        p_length = 0
        if article["paragraphs"] != []:
            for paragraph in article["paragraphs"]:
                p_length += len(paragraph)

        elif article["lead"] != "":
            p_length = len(article["lead"])

        paragraph_length.append(p_length)

        # attribute 2 : group from url 
        url = article["url"]
        url_split = url.split("/")
        group = url_split[3]
        # 'https:', '', 'www.rtvslo.si' so vedno prvi trije, vzamem cetrtega torej
        groups.append(group)

        # attribute 3 : hour
        # attribute 4 : month
        # attribute 5 : year
        # attribute 6 : day of week
        date_string = article["date"]
        dt = datetime.fromisoformat(date_string)

        hour = dt.hour
        month = dt.month
        year = dt.year
        day_of_week = dt.weekday()

        hours.append(hour)
        months.append(month)
        years.append(year)
        days_of_week.append(day_of_week)

        # attribute 7 : author
        if "authors" not in article or article["authors"] == "":
            author = "Unknown"
        else:
            author = article["authors"]
        authors.append(author)



    # transform into panda format
    data_panda = pd.DataFrame({
        "group": groups,
        "hour": hours,
        "month": months,
        "year": years,
        "day_of_week": days_of_week,
        "paragraph_length": paragraph_length,
    })

    # maybe drop nan but idk if theres any
    #print(f"Nan values:\n{data_panda.isna().any()} ")

    # one-hot encode group, hour, month, year, day_of_week
    data_encoded = pd.get_dummies(
        data_panda,
        columns=["group", "hour", "month", "year", "day_of_week"],
        drop_first=False  # set to True if you want to avoid multicollinearity
    )
    #print(data_encoded.head())
    #print(f"Final shape: {data_encoded.shape}")

    #print(f"For now we have following {len(self.attributes)} attributes: {self.attributes}")
    #print(f"target list: {self.target_list}")
    


    # Ensure test has same columns as training
    for col in attributes_list:
        if col not in data_encoded.columns:
            data_encoded[col] = 0  # add missing column

    # Reorder columns to match training
    data_encoded = data_encoded[attributes_list]

    print(f"Loading and processing test data finished.")

    #X_meta = data_encoded.values.astype(np.float32)
    #np.save("TEST_meta_features.npy", X_meta)
    #print(f"Saved metadata features: {X_meta.shape}")

    return data_encoded





BATCH_SIZE = 64

if __name__ == "__main__":

    # prepare data for training
    data = PrepareData("../data/rtvslo_train.json")

    # get X_train and y_train
    X_train, y_train = data.get_train_data()
    print(X_train.shape)

    # send it to nn
    attributes_list = data.get_attributes()
    model = Model_Neural_Network(input_dim=len(attributes_list))
    model.fit(X_train, y_train, batch_size=BATCH_SIZE)

    # Evaluate on test split
    X_test, y_test = data.get_test_data()
    y_pred = model.predict(X_test, attributes_list)

    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    y_test_tensor = y_test.view(-1,1)
    final_mse = nn.L1Loss()(y_pred_tensor, y_test_tensor).item()
    print(f"\nMSE on local test data: {final_mse:.4f}")
  
    # FINAL OUTPUT
    X_final_path = "../data/rtvslo_test.json"
    X_final = prepare_test_data(X_final_path, attributes_list)
    X_final = X_final[attributes_list].values.astype(np.float32)
    
    y_pred_final = model.predict(X_final, attributes_list)

    np.savetxt('TEST_RESULTS_nn.txt', y_pred_final, fmt='%f')

