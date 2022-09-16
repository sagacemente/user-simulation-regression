import sys
import torch
from torch import nn
from torch.nn import Transformer
import time
from sklearn.model_selection import train_test_split
from data_generator.grid_constructor import *
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(torch.nn.Module):
    def __init__(self, sequence_length, num_input_features, num_output_features, num_encoders=3, num_decoders=3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=num_input_features)
        self.Transformer = Transformer(d_model=num_input_features,
                                       num_encoder_layers=num_encoders,
                                       num_decoder_layers=num_decoders)
        self.linear = torch.nn.Linear(sequence_length*num_input_features, num_output_features)

    def forward(self, x_features):
        x_features = self.pos_encoder(x_features)
        x_features = self.Transformer(x_features, x_features)
        x_features = x_features.relu()

        x_features = torch.moveaxis(x_features, 0, 1)
        x_features = torch.reshape(x_features, (x_features.shape[0], x_features.shape[1]*x_features.shape[2]))

        x_features = self.linear(x_features)

        return x_features


start = time.time()
TEST_SIZE = 0.1

x = []
y = []
#all_files = os.listdir('../data/samples')
all_files = os.listdir('../data/samples_clear')
for f in all_files:
    with open('../data/samples_clear/'+f, 'rb') as handle:
        output_dict = pickle.load(handle)
        x.append(output_dict['x'])
        y.append(output_dict['y'])

print('file loaded')
x = np.array(x)
print('x pre shape', x.shape)
x = average_input(x)
x = x.reshape(x.shape[0], x.shape[1], -1)
x = np.moveaxis(x, 0, 1)
# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print('x post shape', x.shape)

y = np.array(y)
y = y[:, 0, :-2]
print('y shape', y.shape)

x_train = x[:, :x.shape[1]-int((x.shape[1]*TEST_SIZE)), :]
x_test = x[:, x.shape[1]-int((x.shape[1]*TEST_SIZE)):, :]
y_train = y[:y.shape[0]-int((y.shape[0]*TEST_SIZE)), :]
y_test = y[y.shape[0]-int((y.shape[0]*TEST_SIZE)):, :]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
for i in [x_train, x_test, y_train, y_test]:
    print('set shape\t', i.shape)


model = TransformerModel(sequence_length=x_train.shape[0],
                         num_input_features=x_train.shape[-1],
                         num_output_features=y_train.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fct = torch.nn.L1Loss()

NUM_EPOCHS = 250
BATCH_SIZE = 16
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = []
    for batch in range(0, x_train.shape[1], BATCH_SIZE):
        if x_train.shape[1] > batch+BATCH_SIZE:
            batch_data = x_train[:, batch:batch+BATCH_SIZE, :]
            batch_labels = y_train[batch:batch+BATCH_SIZE, :]
        else:
            batch_data = x_train[:, batch:, :]
            batch_labels = y_train[batch:, :]
        y_hat = model(torch.FloatTensor(batch_data))
        loss = loss_fct(y_hat, torch.FloatTensor(batch_labels))
        # print(f'epoch {epoch}, batch {batch}, loss {loss}')
        train_loss.append(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'epoch {epoch}, loss {sum(train_loss)/len(train_loss)}')

    model.eval()
    y_hat_all = []
    y_true_all = []
    for batch in range(0, x_test.shape[0], BATCH_SIZE):
        if x_test.shape[0] > batch+BATCH_SIZE:
            batch_data = x_test[:, batch:batch+BATCH_SIZE, :]
            batch_labels = y_test[batch:batch+BATCH_SIZE, :]
        else:
            batch_data = x_test[:, batch:, :]
            batch_labels = y_test[batch:, :]
        y_hat_all.append(model(torch.FloatTensor(batch_data)))
        y_true_all.append(torch.FloatTensor(batch_labels))

    loss = loss_fct(torch.cat(y_hat_all), torch.cat(y_true_all))
    print(f'eval loss {loss}')
