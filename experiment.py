import torch
import torch.nn.functional as F


def test(net, loader, criterion, c):
    tot_loss, samples, correct = 0, 0, 0
    for X, y in loader:
        X, y = X.to(c.device), y.to(c.device)
        output = net(X)
        predicts = F.softmax(output, dim=1)
        correct += torch.sum(torch.argmax(predicts, dim=1) == y).item()
        loss = criterion(output, y)
        tot_loss += loss.item()
        samples += len(X)
    return tot_loss/samples, correct/samples


def learn(net, train_load, val_load, criterion, optim, c):
    for e in range(c.epochs):
        tot_loss, samples = 0, 0
        for X, y in train_load:
            optim.zero_grad()
            X, y = X.to(c.device), y.to(c.device)
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optim.step()
            tot_loss += loss.item()
            samples += len(X)
        print(f'Epoch {e}')
        print(f'Train loss: {tot_loss/samples}')
        v_loss, v_acc = test(net, val_load, criterion, c)
        print(f'Validation loss: {v_loss}')
        print(f'Validation acc: {v_acc}')