"""
Based on implementation from https://github.com/automan000/Convolution_LSTM_pytorch
"""
import torch
from models.convolution_lstm import ConvLSTM


class Model(torch.nn.Module):
    def __init__(self, num_classes=174, nb_lstm_units=32, channels=3, conv_kernel_size=(5, 5), pool_kernel_size=(2, 2),
                 top_layer=True, avg_pool=False, batch_normalization=True, lstm_layers=4, step=16,
                 image_size=(224, 224), dropout=0, conv_stride=(1, 1), effective_step=[4, 8, 12, 15],
                 use_entire_seq=False, add_softmax=False):

        super(Model, self).__init__()

        self.num_classes = num_classes
        self.nb_lstm_units = nb_lstm_units
        self.channels = channels
        self.top_layer = top_layer
        self.avg_pool = avg_pool
        self.c_kernel_size = conv_kernel_size
        self.lstm_layers = lstm_layers
        self.step = step
        self.im_size = image_size
        self.pool_kernel_size = pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.conv_stride = conv_stride
        self.effective_step = effective_step
        self.add_softmax = add_softmax
        self.use_entire_seq = use_entire_seq
        self.clstm = None
        self.endFC = None
        self.sm = None

        self.build()

    def build(self):
        """
        pytorch CLSTM usage: 
        clstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64],
                         kernel_size=5, step=9, effective_step=[2, 4, 8])
        lstm_outputs = clstm(cnn_features)
        hidden_states = lstm_outputs[0]

        """
        self.clstm = ConvLSTM(input_channels=self.channels,
                              hidden_channels=[self.nb_lstm_units] * self.lstm_layers,
                              kernel_size=self.c_kernel_size[0], conv_stride=self.conv_stride,
                              pool_kernel_size=self.pool_kernel_size, step=self.step,
                              effective_step=self.effective_step,
                              batch_normalization=self.batch_normalization, dropout=self.dropout)

        if self.use_entire_seq:
            self.endFC = torch.nn.Linear(in_features=len(self.effective_step) * self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)),
                                         out_features=self.num_classes)
        else:
            self.endFC = torch.nn.Linear(in_features=self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)),
                                         out_features=self.num_classes)

        print("use entire sequence is: ", self.use_entire_seq)
        print("shape of FC is: ", self.endFC)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):

        output, hiddens = self.clstm(x)

        if self.use_entire_seq:
            output = self.endFC(torch.stack(output).view(-1, len(self.effective_step) * self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers))))
        else:
            output = self.endFC(output[-1].view(-1, self.nb_lstm_units * int(
                self.im_size[0] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers)) * int(
                self.im_size[1] / ((self.conv_stride * self.pool_kernel_size[0]) ** self.lstm_layers))))

        if self.add_softmax:
            output = self.sm(output)

        return output

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
