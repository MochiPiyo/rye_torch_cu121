import torch
from torchvision import datasets, transforms

# 2024_02_21
# https://github.com/pytorch/examples/blob/main/mnist/main.py

device = 'cuda'
batch_size = 256
epochs = 14
log_interval = 10

sparsity = 1.0
sparsity_lr = 0.001
sparsity_gravity = 0.05


class MaskedProp():
    def __init__(self, val: torch.Tensor, mask: torch.Tensor) -> None:
        self.val = val
        self.mask = mask

class HiyokoSparse(torch.nn.Module):
    def __init__(self, input: int, output: int, 
                 sparsity: float, sparsity_lr: float, sparsity_gravity: float) -> None:
        super().__init__()
        # 目標の有効辺割合
        self.opjective_sparsity = sparsity
        # learning_rateに相当するもの
        self.sparsity_lr = sparsity_lr
        # 収束力
        self.sparsity_gravity = sparsity_gravity

        self.model_sparsity = 1.0

        self.weight = torch.nn.Parameter(torch.randn(input, output).to(device))
        self.bias = torch.nn.Parameter(torch.zeros(output).to(device))
        self.importances = torch.ones_like(self.weight, dtype=torch.float32).to(device)
        # forward内で生成
        # self.input_mask: Tensor2d<bool, B, IN>
        # self.output_mask: Tensor2d<bool, B, OUT>

    
    def forward(self, mprop: MaskedProp) -> MaskedProp:
        self.input_mask = mprop.mask
        x = mprop.val

        # self.weightのうち、self.importancesが0より小さい位置に相当する場所を0にする
        masked_weight = self.weight.data.clone()
        masked_weight[self.importances < 0] = 0
        x = torch.matmul(x, masked_weight) + self.bias
        
        self.output_mask = x > 0.0
        
        x = torch.nn.functional.relu(x)

        
        #print(">> self.output_mask", self.output_mask.sum().item())
        return MaskedProp(x, self.output_mask)
    
    def update_sparsity(self):
        # true -> 1, false -> 0
        in_fired = self.input_mask.to(torch.float32)
        #print(in_fired)
        # true -> 1, false -> -1 (1 * 2 - 1 = 1, 0 * 2 - 1 = -1である)
        out_fired = self.output_mask.to(torch.float32) * 2 - 1
        #print(out_fired)
        # matmul(Tensor<IN, B>, Tensor<B, OUT>)
        importances_in_this_batch = torch.matmul(in_fired.transpose(0, 1), out_fired)
        # Batch分だけ加算されるから一個あたりにすべき
        importances_in_this_batch /= batch_size
        #print(">> importances_in_this_batch", importances_in_this_batch)

        # 目標のsparcityに近づけるための調整項目
        num_elements = self.importances.numel()
        num_valid_edge = (self.importances > 0).sum().item()
        sparsity_rate_now = num_valid_edge / num_elements
        self.model_sparsity = sparsity_rate_now
        print(">> temp spartisy", sparsity_rate_now)
        # 目標との差分
        sp_rate_diff = (self.opjective_sparsity - sparsity_rate_now) * self.sparsity_gravity
        
        # スカラーのTensorを作ってsigmoid。
        # (-1..1)の範囲に押し込める。gravityで抑制
        balancer = torch.sigmoid(torch.tensor(sp_rate_diff).to(device)) * 2 - 1.0
        print(">> balancer", balancer)
        update = importances_in_this_batch * self.sparsity_lr + balancer
        print(">> update", update)

        self.importances += update
        print(">> importances", self.importances)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.hs1 = HiyokoSparse(28 * 28, 256, sparsity, sparsity_lr, sparsity_gravity)
        self.hs2 = HiyokoSparse(256, 10, sparsity, sparsity_lr, sparsity_gravity)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        #print("input x >>", x)

        # MaskedProp
        x = MaskedProp(x, x > 0)
        #print("input x1 >>", x.val)
        x = self.hs1(x)
        #print("input x2 >>", x.val)
        x = self.hs2(x)
        x = x.val
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output
    
    def update_sparsity(self):
        self.hs1.update_sparsity()
        self.hs2.update_sparsity()

def train(log_interval, model: Net, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print("data >>", data)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        model.update_sparsity()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
            ))
            print("sparsity1: {:.3}, sparsity2: {:.3}".format(model.hs1.model_sparsity, model.hs2.model_sparsity))

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
        # transforms.Normalize((0.1307,), (0.3081))
    ])
    dataset1 = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./mnist', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwrags)

    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
   
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