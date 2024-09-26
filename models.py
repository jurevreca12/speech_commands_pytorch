import torch.nn as nn

def make_ds_block(ch_in, ch_mid, ch_out):
	m = nn.Sequential(
		nn.Conv2d(
			in_channels=ch_in,
			out_channels=ch_mid,
			kernel_size=3,
		 	stride=1,
			padding=0,
			groups=ch_in
		),
        nn.BatchNorm2d(ch_mid),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=ch_mid,
            out_channels=ch_out,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        ),
        nn.BatchNorm2d(ch_out),
        nn.ReLU()
	)
	return m

def make_base_conv(ch_in, ch_out):
    m = nn.Sequential(
        nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size = 3,
            stride = 1,
            padding = 0,
            groups = 1,
        ),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(),
    )
    return m

def get_model():
    class ConvModel(nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            # in 1 x 20 x 32
            self.conv1 = make_base_conv(1, 16) # 16 x 18 x 30
            self.ds_conv1 = make_ds_block(16, 32, 32)  # 16 x 16 x 28
            self.ds_conv2 = make_ds_block(32, 64, 8)   # 8 x 14 x 26
            self.maxpool = nn.MaxPool2d(2)  # 8 x 7 x 13
            self.linear = nn.Linear(8*7*13, 12)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.ds_conv1(x)
            x = self.ds_conv2(x)
            x = self.maxpool(x)
    
            # flatten the output for linear layer
            x = x.view(x.size(0), -1)
            out =  self.linear(x)
            return out
    
    return ConvModel()

