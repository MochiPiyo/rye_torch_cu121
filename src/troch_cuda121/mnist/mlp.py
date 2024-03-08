import torch
from torchvision import datasets, transforms

# 2024_02_21
# https://github.com/pytorch/examples/blob/main/mnist/main.py


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 200)
        self.fc2 = torch.nn.Linear(200, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            # sum up batch loss
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def main():
    device = torch.device("cuda")

    # ker word agrument、辞書形式で引数を与える
    train_kwargs = {'batch_size': 256}
    test_kwrags = {'batch_size': 1000 }
    cuda_kwags = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True,
    }
    train_kwargs.update(cuda_kwags)
    test_kwrags.update(cuda_kwags)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # (平均, 標準偏差)
        transforms.Normalize((0.1307,), (0.3081))
    ])
    dataset1 = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./mnist', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwrags)

    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    epochs = 14
    log_interval = 10
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    """
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    """

if __name__ == '__main__':
    main()