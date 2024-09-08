import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# Parameters
ROOT_DIR = '.'
DATA_DIR = 'data'
DATA_DIR = os.path.join(ROOT_DIR, DATA_DIR)
SEED = 20244078
data_name = "korea-clean"
BATCH_SIZE = 32
LUCKY_NUMBER = 42
lr = 1e-3
epochs = 200
beta1, beta2 = 0.9, 0.999
criterion = nn.MSELoss()

print(f"Data directory: {DATA_DIR}")
print(f"Seed: {SEED}")
print(f"Data name: {data_name}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv(os.path.join(DATA_DIR, f"{data_name}.csv"))
# df.head(5)
# df.info()

i = 0
for nan in df.isna().sum():
    if nan > 0:
        # print(nan)
        i += 1
print(f"Number of columns with NaN: {i}")
        
plt.figure(figsize=(100,100))
df_corr_matrix = df.corr()
sns.heatmap(df_corr_matrix, annot = True, fmt ='.2f', linewidth = .7)
plt.title(f'Relationship Between Features for {data_name}')
plt.savefig(os.path.join(DATA_DIR, f"corr-{data_name}.png"))
# plt.show()
print(f"Correlation matrix save path: {os.path.join(DATA_DIR, f'corr-{data_name}.png')}")

correlation_with_happiness = df_corr_matrix['happiness_ladder']
print()
print(correlation_with_happiness)
print("\n==========\n")

print(f"Batch size: {BATCH_SIZE}")
df_train = pd.read_csv(os.path.join(DATA_DIR, f"{data_name}-train.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, f"{data_name}-test.csv"))

df_train['bias'] = -1.
df_test['bias'] = -1.

if 'shuffle' not in data_name:
    del df_train['year']
    del df_test['year']

df_train_label = torch.tensor(df_train['happiness_ladder'].tolist(), dtype=torch.float).to(device)
df_test_label = torch.tensor(df_test['happiness_ladder'].tolist(), dtype=torch.float).to(device)

del df_train['happiness_ladder']
del df_test['happiness_ladder']

df_train_data = torch.tensor(df_train.values, dtype=torch.float).to(device)
df_test_data = torch.tensor(df_test.values, dtype=torch.float).to(device)

df_train_data.shape, df_train_label.shape, df_test_data.shape, df_test_label.shape

train_dataset = torch.utils.data.TensorDataset(df_train_data, df_train_label)
test_dataset = torch.utils.data.TensorDataset(df_test_data, df_test_label)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

all_in = df_train_data.shape[1]
print(f'Number of input features: {all_in}')

class Regressor(nn.Module):
    def __init__(self, features):
        super(Regressor, self).__init__()


        self.layers = []
        for i in range(len(features)-1):
            self.layers.append(nn.Linear(features[i], features[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(features[-1], 1))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

print(f"Lucky number: {LUCKY_NUMBER}")
print(f"Epochs: {epochs}")
print(f"Adam betas: {beta1}, {beta2}")
print(f"Criterion: MSE loss\n")

class Model:
    def __init__(self, features, lr, beta1, beta2, epochs):
        self.features = features
        self.model = Regressor(self.features)
        self.model = self.model.to(device)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.train_loss_epoch = None
        self.test_err_epoch = None

    def update(self, train_loss_epoch, test_err_epoch):
        self.train_loss_epoch = train_loss_epoch
        self.test_err_epoch = test_err_epoch
        
models = [  Model([all_in, LUCKY_NUMBER, LUCKY_NUMBER*2, LUCKY_NUMBER*4, LUCKY_NUMBER*8, LUCKY_NUMBER*16, LUCKY_NUMBER*8, LUCKY_NUMBER*4, LUCKY_NUMBER*2, LUCKY_NUMBER], lr, beta1, beta2, epochs),
            Model([all_in, LUCKY_NUMBER, LUCKY_NUMBER*2, LUCKY_NUMBER*4, LUCKY_NUMBER*8, LUCKY_NUMBER*4, LUCKY_NUMBER*2, LUCKY_NUMBER], lr, beta1, beta2, epochs),
            Model([all_in, LUCKY_NUMBER, LUCKY_NUMBER*2, LUCKY_NUMBER*4, LUCKY_NUMBER*2, LUCKY_NUMBER], lr, beta1, beta2, epochs),
            Model([all_in, LUCKY_NUMBER, LUCKY_NUMBER*2, LUCKY_NUMBER], lr, beta1, beta2, epochs),
            Model([all_in, LUCKY_NUMBER], lr, beta1, beta2, epochs) ]

def train(model, trainloader, criterion):

    model.model.train()
    losses = []

    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        model.optimizer.zero_grad()
        output = model.model(data)
        loss = criterion(output, label.reshape(-1,1))
        losses.append(loss)
        loss.backward()
        model.optimizer.step()

    loss_epoch = sum(losses)/len(losses)
    return loss_epoch.item()

def eval(model, testloader, show=False):

    model.model.eval()

    sums = []
    labels = []
    outputs = []
    with torch.no_grad():
        for i, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device)
            output = model.model(data)
            squared_error = (output - label.reshape(-1,1))**2
            sum_squared_error = torch.sum(squared_error)
            sums.append(sum_squared_error)
            if show:
                labels += label.tolist()
                outputs += [elem[0] for elem in output.tolist()]

    error_epoch = sum(sums)/len(sums)
    result = pd.DataFrame({'label': labels, 'output': outputs})
    return error_epoch.item(), result

for i in range(len(models)):
    print(f"\nTraining for {data_name} dataset with features={models[i].features}, lr={models[i].lr}, beta1={models[i].beta1}, beta2={models[i].beta2}, epochs={models[i].epochs}")
    train_loss_epoch = []
    test_err_epoch = []
    show = False
    for epoch in range(models[i].epochs):
        if epoch == models[i].epochs-1:
            show = True
        train_loss_epoch.append(train(models[i], train_loader, criterion))
        test_err, result = eval(models[i], test_loader, show=show)
        test_err_epoch.append(test_err)
        if (epoch+1) % 20 == 0:
            print(f'Epoch: {epoch+1}\tTrain Loss: {train_loss_epoch[-1]:5.2f}\tTest SSE: {test_err_epoch[-1]:7.2f}')
            torch.save(models[i].model.state_dict(), os.path.join(ROOT_DIR, f'model/{data_name}-{len(models[i].features)}-{epoch+1}.pth'))
        if show:
            result.to_csv(os.path.join(DATA_DIR, f'inference-{data_name}-{len(models[i].features)}-{epoch+1}.csv'), index=False)
    models[i].update(train_loss_epoch, test_err_epoch)
    
def plot():
    figure = plt.figure(figsize=(12,4))

    f1 = figure.add_subplot(121)
    f2 = figure.add_subplot(122)

    for model in models:
        epochs = range(1, model.epochs+1)
        f1.plot(epochs, model.train_loss_epoch, label=f'{data_name}, {len(model.features)} layers')
    f1.set_title(f'Training loss (MSE Loss)')
    f1.set_xlabel('Epochs')
    f1.set_ylabel('Training Loss')
    f1.legend()

    for model in models:
        f2.plot(epochs, model.test_err_epoch, label=f'{data_name}, {len(model.features)} layers')
    f2.set_title(f'SSE on testing dataset')
    f2.set_xlabel('Epochs')
    f2.set_ylabel('Test SSE')
    f2.legend()
    
    plt.savefig(os.path.join(DATA_DIR, f"train-{data_name}.png"))
    # plt.show()
    print()
    for model in models:
        print(f"SSE on testing dataset for {data_name} with {len(model.features)} layers:\t{model.test_err_epoch[-1]:.4f}")
        
plot()