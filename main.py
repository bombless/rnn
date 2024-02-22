from data import Corpus
from model import RNN
import torch
import torch.nn as nn

data = Corpus('.')
train_data = data.train
test_data = data.test

learning_rate = 0.005

rnn = RNN(1, 128, 2)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


def train(line_pair):
    line_tensor, label = line_pair
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    for idx in range(len(line_tensor)):
        item_tensor = torch.tensor(line_tensor[idx], dtype=torch.int).unsqueeze(0)
        print('item_tensor', item_tensor.shape)
        print('hidden', hidden.shape)
        output_this, hidden = rnn(item_tensor, hidden)
        loss_this = criterion(output_this, torch.tensor(label))
        loss_this.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return output_this, loss_this.item()


for i, line_pair in enumerate(train_data):
    output, loss = train(line_pair)
    if (i + 1) % 100 == 0:
        print(loss)
