    import torch
    import torch.nn as nn
    import numpy as np
    import random

    def set_seed(seed=42):
        """Set the seed for reproducibility."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed()  # Set the seed at the beginning of the script

    # Define the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available!")

    class ConvLSTM(nn.Module):
        """
        A Convolutional Neural Network (CNN) followed by a Long Short-Term Memory (LSTM) network.

        Args:
            in_channels (int): Number of input channels for the convolution layer.
            lstm_input_size (int): Input size for the LSTM layer.
            lstm_hidden_size (int): Hidden size for the LSTM layer.
            lstm_num_layers (int): Number of layers in the LSTM.
            output_size (int): Output size of the fully connected layer.
        """

        def __init__(self, in_channels, lstm_input_size, lstm_hidden_size, lstm_num_layers, output_size):
            super(ConvLSTM, self).__init__()
            self.lstm_hidden_size = lstm_hidden_size
            self.lstm_num_layers = lstm_num_layers
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
            self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
            self.fc = nn.Linear(lstm_hidden_size, output_size)

        def forward(self, x):
            x = self.conv(x)
            batch_size = x.size(0)
            h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
            c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])  # Use the last time step's output as the prediction

            return out
