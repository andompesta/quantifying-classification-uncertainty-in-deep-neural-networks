from src.conf import BaseConf

class LeNetConf(BaseConf):
    def __init__(
            self,
            conv1_in=1,
            conv1_out=20,
            conv1_size=(5, 5),
            conv2_in=20,
            conv2_out=50,
            conv2_size=(5, 5),
            flatten_dim=20000,
            hidden_size=500,
            num_labels=10,
            max_pool1_size=1,
            max_pool2_size=1,
            dropout=.5,
            **kwargs

    ):
        super(LeNetConf, self).__init__("LeNet")
        self.conv1_in = conv1_in
        self.conv1_out = conv1_out
        self.conv1_size = conv1_size
        self.conv2_in = conv2_in
        self.conv2_out = conv2_out
        self.conv2_size = conv2_size
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = dropout
        self.max_pool1_size = max_pool1_size
        self.max_pool2_size = max_pool2_size

        for n, v in kwargs.items():
            setattr(self, n, v)