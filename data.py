import torch
import numpy as np
import tensorflow_datasets as tfds

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

def get_loaders(batch_size):
	train_ds, info = tfds.load(
	    'speech_commands', 
	    split='train',
	    batch_size=batch_size,
	    shuffle_files=True,
	    with_info=True
	)	
	val_ds = tfds.load(
	    'speech_commands', 
	    split='validation',
	    batch_size=batch_size,
	)
	test_ds = tfds.load(
	    'speech_commands', 
	    split='test',
	    batch_size=batch_size,
	)

	return (train_ds, val_ds, test_ds)


def data_classes():
    x = {
      0: "down",
      1: "go",
      2: "left",
      3: "no",
      4: "off",
      5: "on",
      6: "right",
      7: "stop",
      8: "up",
      9: "yes",
      10: "background",
      11: "other"
    }
    return x
