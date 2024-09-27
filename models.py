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

def make_base_conv(ch_in, ch_out, kernel_size):
    m = nn.Sequential(
        nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size = kernel_size,
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
            self.conv1 = make_base_conv(1, 172, 3) # 172 x 18 x 30
            self.ds_conv1 = make_ds_block(172, 172, 172)  # 172 x 16 x 28
            self.ds_conv2 = make_ds_block(172, 172, 172)   # 172 x 14 x 26
            self.ds_conv3 = make_ds_block(172, 172, 172)   # 172 x 12 x 24
            self.maxpool = nn.MaxPool2d(2)  
            self.linear = nn.Linear(172*6*12, 12)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.ds_conv1(x)
            x = self.ds_conv2(x)
            x = self.ds_conv3(x)
            x = self.maxpool(x)
    
            # flatten the output for linear layer
            x = x.view(x.size(0), -1)
            out =  self.linear(x)
            return out
    
    return ConvModel()

