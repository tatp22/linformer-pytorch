import os
import torch
import torch.nn as nn

from collections import OrderedDict
from linformer_pytorch import LinformerLM, Padder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import torchtext
from torchtext.data.utils import get_tokenizer

config = OrderedDict(
    batch_size=16,
    lr=0.1,
    no_cuda=True,
    num_epochs=30,
    output_dir="./output",
    seed=2222,

    seq_len=35,
    ch=16,
    num_tokens=28785, # The output of len(TEXT.vocab.stoi)
    bptt=35,
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

    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token="<sos>",
                                eos_token="<eos>",
                                lower=True)
    train_text, _ ,test_text = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_text)

    train_data = batchify(TEXT, device, train_text, config["batch_size"])
    test_data = batchify(TEXT, device, test_text, config["batch_size"])

    optimizer = get_optimizer(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        print("Epoch {}".format(epoch))

        model.train()
        train_loss = 0

        for batch, i in tqdm(enumerate(range(0, train_data.size(0)-1, config["bptt"]))):
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction.reshape(-1, config["num_tokens"]), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_data)
        print("Training loss: {}".format(train_loss))

        with torch.no_grad():
            model.eval()
            test_loss = 0
            for i in tqdm(range(0, test_data.size(0)-1, config["bptt"])):
                data, targets = get_batch(test_data, i)
                prediction = model(data)
                loss = criterion(prediction.reshape(-1, config["num_tokens"]), targets)
                test_loss  += loss.item()

            test_loss /= len(test_data)
            print("Training loss: {}".format(test_loss))

def get_model(device):
    """
    Gets the device that the model is running on. Currently running standard linformer
    """
    model = LinformerLM(config["num_tokens"], input_size=config["seq_len"], channels=config["ch"], dim_k=64,dim_ff=64, nhead=4, depth=4, activation="gelu", checkpoint_level="C0")
    model = Padder(model)
    model.to(device)
    return model

def get_optimizer(model_parameters):
    """
    Gets an optimizer. Just as an example, I put in SGD
    """
    return torch.optim.SGD(model_parameters, lr=config["lr"])

def batchify(TEXT, device, data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0,0,nbatch*bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(config["bptt"], len(source)-1-i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

if __name__ == "__main__":
    main()
