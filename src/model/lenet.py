import torch
from torch.nn import functional as F
from .utils import BaseModel

class LeNet(BaseModel):
    def __init__(self, conf):
        super(LeNet, self).__init__(conf)

        self.conv1 = torch.nn.Conv2d(
                in_channels=conf.conv1_in,
                kernel_size=conf.conv1_size,
                out_channels=conf.conv1_out
            )

        self.conv2 = torch.nn.Conv2d(
            in_channels=conf.conv2_in,
            out_channels=conf.conv2_out,
            kernel_size=conf.conv2_size
        )

        self.fc1 = torch.nn.Linear(
            conf.flatten_dim, conf.hidden_size
        )

        self.fc2 = torch.nn.Linear(
            conf.hidden_size,
            conf.num_labels
        )

        self.dropout = torch.nn.Dropout(conf.dropout)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=self.conf.max_pool1_size)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=self.conf.max_pool2_size)

        x = torch.reshape(x, (x.shape[0], -1))

        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)

        return logits

    def _init_weights(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
            torch.nn.init.constant_(module.bias, 0.)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
            torch.nn.init.constant_(module.bias, 0.)
        else:
            print(module)


