import torch
import numpy as np
import tensorflow_datasets as tfds
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#def collate_fn(batch):
#    COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
#    waveforms = []
#    labels = []
#
#    for elem in batch:
#        waveform = elem[0]
#        waveform_pad = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[-1]))
#        waveforms.append(waveform_pad)
#        if elem[2] in COMMANDS:
#            labels.append(COMMANDS.index(elem[2]))
#        else:
#            labels.append(10)
#    return torch.from_numpy(np.array(waveforms)), torch.Tensor(labels).to(torch.long)

def get_loaders(batch_size, num_workers):
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
	train_dl = tf_dataset_to_pytorch_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
	val_dl = tf_dataset_to_pytorch_dataloader(val_ds, batch_size, shuffle=False, num_workers=num_workers)
	test_dl = tf_dataset_to_pytorch_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
	return (train_dl, val_dl, test_dl)


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

def iter_tf_data(dataset):
    x_list = []
    y_list = []
    for data in dataset.as_numpy_iterator():
        x = data['audio'].astype(np.float32)
        y = data['label']
        x_list += [torch.from_numpy(x)] 
        y_list += [torch.from_numpy(y)]
    x_list_cat = torch.cat(x_list, axis=0)
    x_list_cat = torch.unsqueeze(x_list_cat, dim=1)
    y_list_cat = torch.cat(y_list, axis=0)
    return [x_list_cat, y_list_cat]

def tf_dataset_to_pytorch_dataloader(
    tf_dataset, batch_size, shuffle=True, num_workers=0
):
    """Converts a TensorFlow Dataset to a PyTorch DataLoader."""
    data_list = iter_tf_data(tf_dataset)
    pytorch_dataset = TensorDataset(*data_list)
    pytorch_dataloader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return pytorch_dataloader
