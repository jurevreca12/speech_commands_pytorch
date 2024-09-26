import torch.nn as nn

def make_ds_block(filters_in, filters_out):
	m = nn.Sequential(
		nn.Conv2d(
			in_channels=filters_in,
			out_channels=filters_out,
			kernel_size=3,
		 	stride=1,
			padding=0,
			groups=filters_out
		)
	)
	return m


def get_model():
    class ConvModel(nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            
            self.conv1 = nn.Sequential(
                nn.Conv2d( # 1 x 20 x 32
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    groups=1,
                ),
                #nn.BatchNorm2d(),
                nn.ReLU(), # 16 x 18 x 30
            )
    
            self.ds_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    groups=16
                ),
                #nn.BatchNorm2d(),
                nn.ReLU(),  # 32 x 16 x 28
                nn.Conv2d(
                    in_channels=32,
                    out_channels=8,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1
                ),
                #nn.BatchNorm2d(),
                nn.ReLU(),  # 8 x 16 x 28
            )
            self.maxpool = nn.MaxPool2d(2)
            self.linear = nn.Linear(8*8*14, 12)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.ds_conv(x)
            x = self.maxpool(x)
    
            # flatten the output for linear layer
            x = x.view(x.size(0), -1)
            out =  self.linear(x)
            return out
    
    return ConvModel()

