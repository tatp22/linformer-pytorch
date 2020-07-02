import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "../")
from collections import OrderedDict
from linformer_pytorch import Linformer, Visualizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

config = OrderedDict(
    batch_size=16,
    lr=0.1,
    no_cuda=True,
    num_epochs=30,
    output_dir="./output",
    seed=2222,

    dummy_seq_len=64,
    dummy_ch=16,
)

output_dir = "./output"

def main():
    """
    Train a model
    """
    global output_dir
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if not config["no_cuda"] and torch.cuda.is_available() else "cpu")

    model = get_model(device)

    training_loader = get_loader()
    test_loader = get_loader(test=True)

    optimizer = get_optimizer(model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(config["num_epochs"]):
        print("Epoch {}".format(epoch))

        model.train()
        train_loss = 0

        for batch_x, batch_y in tqdm(training_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(training_loader)
        print("Training loss: {}".format(train_loss))

        with torch.no_grad():
            model.eval()
            test_loss = 0
            for batch_x, batch_y in tqdm(test_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                test_loss  += loss.item()

            test_loss /= len(test_loader)
            print("Testing loss: {}".format(test_loss))

def get_model(device):
    """
    Gets the device that the model is running on. Currently running standard linformer
    """
    model = Linformer(input_size=config["dummy_seq_len"], channels=config["dummy_ch"], dim_d=config["dummy_ch"], dim_k=64,dim_ff=64, nhead=4, depth=2, activation="gelu", checkpoint_level="C0", full_attention=True, include_ff=False, parameter_sharing="none")
    model.to(device)
    return model

def get_optimizer(model_parameters):
    """
    Gets an optimizer. Just as an example, I put in SGD
    """
    return torch.optim.SGD(model_parameters, lr=config["lr"])

def get_loader(test=False):
    """
    Gets data and a loader. Just dummy data, but replace with what you want
    """
    data_points=128
    if test:
        data_points=32

    x_tensor = torch.randn((data_points, config["dummy_seq_len"], config["dummy_ch"]), dtype=torch.float32)
    y_tensor = torch.randn((data_points, config["dummy_seq_len"], config["dummy_ch"]), dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=config["batch_size"], num_workers=2)

if __name__ == "__main__":
    main()
