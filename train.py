import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS as SpeechCommands
import numpy as np
import argparse
from torchinfo import summary
from data import get_loaders
from models import get_model


def get_parser():
    parser = argparse.ArgumentParser(
                    prog='speech_commands_pytorch',
                    description='Trains a model on the speech comands dataset.')
    parser.add_argument('-n', '--num-epochs', type=int, help='Number of epochs to train the model', default=10)
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('-w', '--num-workers', type=int, help='Number of workers for preprocessing.', default=4)
    parser.add_argument('-p', '--pin-memory', action='store_true', default=False)
    return parser

def evaluate(model, loader, transform, loss_fn):
    model.eval()
    size = len(loader.dataset)
    num_batches = len(loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for waveforms, predictions in loader:
            mel_spectrograms = transform(waveforms)
            outputs = model(mel_spectrograms)
            test_loss += loss_fn(outputs, predictions).item()
            correct += (outputs.argmax(1) == predictions).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_loop(model, train_loader, transform, optimizer, loss_fn):
    model.train()
    for i, (waveforms, predictions) in enumerate(train_loader):
        # zero gradients from previous loop
        optimizer.zero_grad()
    
        # we trasfrom the waveforms into spectorgrams and feed them into the model
        mel_spectrograms = transform(waveforms)
        outputs = model(mel_spectrograms)
        loss = loss_fn(outputs, predictions)
    
        # calculate losses and step the optimizer
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f"Iteration: [{i+1}/{len(train_loader)}]; Loss: {loss:.4f}") 
 

def train(args):
    train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.num_workers, args.pin_memory)
    transform = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        hop_length=512,
        n_mels=20
    )

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(summary(model, (1, *(1, 20, 32))))
    model.train()
    for epoch in range(args.num_epochs):
        print(f"EPOCH: [{epoch+1}/{args.num_epochs}]") 
        train_loop(model, train_loader, transform, optimizer, loss_fn)
        print("Val error:")
        evaluate(model, val_loader, transform, loss_fn)

    print("Test error:")
    evaluate(model, test_loader, transform, loss_fn)
if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    train(args)