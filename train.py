import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS as SpeechCommands
import numpy as np
import argparse
from torchinfo import summary
from data import get_loaders, data_classes
from models import get_model
from torcheval.metrics import MulticlassAccuracy
import wandb

def get_parser():
    parser = argparse.ArgumentParser(
                    prog='speech_commands_pytorch',
                    description='Trains a model on the speech comands dataset.')
    parser.add_argument('-n', '--num-epochs', type=int, help='Number of epochs to train the model', default=20)
    parser.add_argument('-w', '--num-workers', type=int, help='Number of workers for dataloading', default=4)
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=256)
    parser.add_argument('-lr', '--learning-rate', type=float, help='Learning rate for training.', default=0.0001)
    parser.add_argument('-fft', '--num-fft', type=float, help='FFT size.', default=512)
    parser.add_argument('-m', '--num-mels', type=float, help='Number of mels.', default=20)
    return parser

def pprint_class_acc(x):
    txt = 'Class Accuracies:\n'
    classes = data_classes()
    for val, cls in classes.items():
        txt += f'{cls} : {x[val]}\n'
    txt += '-----------------\n'
    print(txt)

def evaluate(model, dataloader, transform, loss_fn, device, name=''):
    model.eval()
    size = 0
    num_batches = 0
    test_loss, correct = 0, 0
    metric = MulticlassAccuracy(average=None, num_classes=12)

    with torch.no_grad():
        for waveforms, labels in dataloader:
            num_batches += 1
            size += len(labels)
            mel_spectrograms = transform(waveforms.to(device))
            outputs = model(mel_spectrograms)
            test_loss += loss_fn(outputs, labels.to(device)).item()
            correct += (outputs.argmax(1) == labels.to(device)).type(torch.float).sum().item()
            metric.update(outputs, labels)
        res = metric.compute()

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    pprint_class_acc(res)
    wandb.log({
        f'{name}_loss'           : test_loss,
        f'{name}_acc'            : correct,
        f'{name}_acc_down'       : res[0],
        f'{name}_acc_go'         : res[1],
        f'{name}_acc_left'       : res[2],
        f'{name}_acc_no'         : res[3],
        f'{name}_acc_off'        : res[4],
        f'{name}_acc_on'         : res[5],
        f'{name}_acc_right'      : res[6],
        f'{name}_acc_stop'       : res[7],
        f'{name}_acc_up'         : res[8],
        f'{name}_acc_yes'        : res[9],
        f'{name}_acc_background' : res[10],
        f'{name}_acc_other '     : res[11],
    })

def train_loop(model, train_dl, transform, optimizer, loss_fn, device):
    model.train()
    for i, (waveforms, labels) in enumerate(train_dl):
        # zero gradients from previous loop
        optimizer.zero_grad()
    
        # we trasfrom the waveforms into spectorgrams and feed them into the model
        mel_spectrograms = transform(waveforms.to(device))
        outputs = model(mel_spectrograms)
        loss = loss_fn(outputs, labels.to(device))
    
        # calculate losses and step the optimizer
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f"Iteration: [{i+1}/{len(train_dl)}]; Loss: {loss:.4f}")
            wandb.log({'train_loss' : loss}) 
 

def train(args):
    train_dl, val_dl, test_dl = get_loaders(args.batch_size, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TRAINING ON: {device}.")
    
    transform = torchaudio.transforms.MelSpectrogram(
        n_fft=args.num_fft,
        hop_length=args.num_fft,
        n_mels=args.num_mels
    ).to(device)

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1] * 11 + [0.1]).to(device))   # The  last class "other" is most common

    with torch.no_grad():
        input_shape = (args.batch_size,) + transform(torch.randn(1, 16000, device=device)).shape
    print(summary(model, input_shape))
    model.train()
    for epoch in range(args.num_epochs):
        print(f"EPOCH: [{epoch+1}/{args.num_epochs}]") 
        train_loop(model, train_dl, transform, optimizer, loss_fn, device)
        print("Val error:")
        evaluate(model, val_dl, transform, loss_fn, device, 'validation')

    print("Test error:")
    evaluate(model, test_dl, transform, loss_fn, device, 'test')


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    wandb.init(
        project="speech_commands_pytorch",

        # track hyperparameters and run metadata
        config={
            "architecture": "DS-CNN",
            "dataset": "SpeechCommands",
            **vars(args)
        }
    )

    train(args)
