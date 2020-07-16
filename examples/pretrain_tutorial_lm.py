import math
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
    gamma=0.95,
    log_interval=100,
    lr=5.0,
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

    optimizer, scheduler = get_optimizer(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        print("Epoch {}".format(epoch+1))

        model.train()
        train_loss = 0
        logging_loss = 0

        for batch, i in tqdm(enumerate(range(0, train_data.size(0)-1, config["bptt"]))):
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction.reshape(-1, config["num_tokens"]), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
            logging_loss += loss.item()

            if batch % config["log_interval"] == 0 and batch > 0:
                curr_loss = logging_loss / config["log_interval"]
                print("Epoch: {}, LR: {}, loss: {}, ppl: {}".format(epoch+1, scheduler.get_last_lr()[0], curr_loss, math.exp(curr_loss)))
                logging_loss = 0

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

            test_loss /= (len(test_data)-1)
            print("Epoch: {}, test_loss: {}, test_ppl: {}".format(epoch+1, test_loss, math.exp(test_loss)))

        scheduler.step()

def get_model(device):
    """
    Gets the device that the model is running on. Currently running standard linformer
    """
    model = LinformerLM(config["num_tokens"], input_size=config["seq_len"], channels=config["ch"], dim_k=35,dim_ff=200, nhead=2, depth=2, activation="gelu", checkpoint_level="C0")
    model = Padder(model)
    model.to(device)
    return model

def get_optimizer(model_parameters):
    """
    Gets an optimizer. Just as an example, I put in SGD
    """
    optim = torch.optim.SGD(model_parameters, lr=config["lr"])
    sched = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=config["gamma"])
    return optim, sched

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
