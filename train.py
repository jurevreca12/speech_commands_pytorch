import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS as SpeechCommands
import numpy as np


class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d( # 1 x 20 x 32
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(), # 16 x 18 x 30
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),  # 32 x 16 x 28
        )
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(32*8*14, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        # flatten the output for linear layer
        x = x.view(x.size(0), -1)
        out =  self.linear(x)
        return out


def collate_fn(batch):
    COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    waveforms = []
    labels = []

    for elem in batch:
        waveform = elem[0]
        waveform_pad = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[-1]))
        waveforms.append(waveform_pad)
        if elem[2] in COMMANDS:
            labels.append(COMMANDS.index(elem[2]))
        else:
            labels.append(10)
    return torch.from_numpy(np.array(waveforms)), torch.Tensor(labels).to(torch.long)

def train():
    NUM_WORKERS = 4
    PIN_MEMORY = False
    BATCH_SIZE = 128
    NUM_EPOCHS = 1

    train_ds = SpeechCommands(root="./data/", download=True, subset="training")
    val_ds = SpeechCommands(root="./data/", download=True, subset="validation")
    test_ds = SpeechCommands(root="./data/", download=True, subset="testing")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    transform = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        hop_length=512,
        n_mels=20
    )

    model = SimpleConvModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
   
    model.train()
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader):
            # get data
            waveforms, predictions = data

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
                print(f"Epoch: [{epoch}/{NUM_EPOCHS}]; Iteration: [{i+1}/{len(train_loader)}]; Loss: {loss:.4f}")


if __name__ == "__main__":
    train()
