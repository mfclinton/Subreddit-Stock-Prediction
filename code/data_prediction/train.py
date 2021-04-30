import torch
import pandas as pd
import numpy
from models.Prelim_Model import *
import os
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train(data):
    # Data Processing
    features = data.to_numpy()[:,:-1]
    labels = data.to_numpy()[:,-1]
    print("Goes Up Percentage = ", labels.sum() / labels.shape[0])
    num_features = features.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    X_train = torch.from_numpy(X_train).cuda()
    X_test = torch.from_numpy(X_test).cuda()
    y_train = torch.from_numpy(y_train).cuda()
    y_test = torch.from_numpy(y_test).cuda()

    train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    # test_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    model = Prelim_Model(num_features, 20, 1).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.BCELoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            predicted_labels = torch.round(outputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_accuracy += (predicted_labels == labels).sum() / labels.size()[0]
            # print("Loss : ", loss)
        print("Avg Loss : ", total_loss / i)
        print("Avg Accuracy : ", total_accuracy / i)

    outputs = model(X_test).squeeze()
    predicted_labels = torch.round(outputs)
    accuracy = (predicted_labels == y_test).sum() / y_test.size()[0]
    print("Test Accuracy : ", accuracy)








if __name__ == "__main__":
    for filename in os.listdir("data"):
        if not filename.endswith("_features_labels.csv"):
            continue
        print(f"--------------{filename}-------------")
        data = pd.read_csv(f"data/{filename}")
        train(data)
