import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from models import LeNet, CustomizedLeNet


def training(trainloader, net, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum)

    net.train()
    for epoch in range(config.nb_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'Epoch {epoch} loss: {running_loss / i}')
        running_loss = 0.0

    print('Finished Training')


def validation():
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            batch_size = images.shape[0]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def save_model(net, filepath):
    torch.save(net.state_dict(), filepath)


class Config():
    def __init__(self, nb_epochs, learning_rate, momentum):
        nb_epochs = nb_epochs
        lr = learning_rate
        momentum = momentum


if __name__ == "__main__":
    saved_model_path = None
    if saved_model_path:
        net.load_state_dict(torch.load(saved_model_path))
    else:
        net = CustomizedLeNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    trainloader, testloader = get_loaders()
    config = Config(10, 0.001, 0.9)
    training(trainloader, net, Config)
    validation(net)
