import torch
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS as SpeechCommands

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

def get_loaders(batch_size, num_workers, pin_memory):
	train_ds = SpeechCommands(root="./data/", download=True, subset="training")
	val_ds = SpeechCommands(root="./data/", download=True, subset="validation")
	test_ds = SpeechCommands(root="./data/", download=True, subset="testing")
	
	train_loader = torch.utils.data.DataLoader(
	    train_ds,
	    batch_size=batch_size,
	    shuffle=True,
	    collate_fn=collate_fn,
	    num_workers=num_workers,
	    pin_memory=pin_memory,
	)
	val_loader = torch.utils.data.DataLoader(
	    val_ds,
	    batch_size=batch_size,
	    shuffle=False,
	    collate_fn=collate_fn,
	    num_workers=num_workers,
	    pin_memory=pin_memory,
	)
	test_loader = torch.utils.data.DataLoader(
	    test_ds,
	    batch_size=batch_size,
	    shuffle=False,
	    collate_fn=collate_fn,
	    num_workers=num_workers,
	    pin_memory=pin_memory,
	)
	return (train_loader, val_loader, test_loader)
