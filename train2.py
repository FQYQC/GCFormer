import tqdm
import torch
from torch.utils.data import dataset, DataLoader
import dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import TransformerModel2
import wandb

def train(epoch, model, dataloader, criterion, opt):
    tot_accnum = 0
    totnum = 0
    model.train()
    for (data, label, mask, diff) in tqdm.tqdm(dataloader):
        data, label = data.cuda(), label.cuda()
        mask = mask.cuda()
        diff = diff.cuda()
        opt.zero_grad()
        output = model(data.permute(1,0), diff.permute(1,0), src_key_padding_mask = mask)
        output = output.permute(1,2,0)
        l = criterion(output, label)
        l.backward()
        opt.step()
        
        accnum = ((label>0)*(torch.argmax(output, 1)==label)).sum()
        tot_accnum += accnum
        totnum += (label>0).sum()
        
        wandb.log({
            "epoch":epoch,
            "train_loss":l,
            "accuracy":accnum/((label>0).sum()),
            "avg_acc":tot_accnum/totnum
        })

def test(epoch, model, dataloader, criterion):
    tot_accnum = 0
    totnum = 0
    model.eval()
    for (data, label, mask, diff) in tqdm.tqdm(dataloader):
        data, label = data.cuda(), label.cuda()
        mask = mask.cuda()
        diff = diff.cuda()
        output = model(data.permute(1,0), diff.permute(1,0), src_key_padding_mask = mask)
        output = output.permute(1,2,0)
        l = criterion(output, label)
        
        accnum = ((label>0)*(torch.argmax(output, 1)==label)).sum()
        tot_accnum += accnum
        totnum += (label>0).sum()
        
        print(f"epoch:{epoch}  testloss:{l:.3f}  acc:{accnum/((label>0).sum()):.3f}  avg_acc:{tot_accnum/totnum:.3f}")

    wandb.log({
        "test_accuracy":tot_accnum/totnum
    })

name = "traindiff1"
torch.set_printoptions(edgeitems=torch.inf)
wandb.init(
    # set the wandb project where this run will be logged
    project="aimusic",
    name = name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "batchsize": 64,
    "epochs": 100,
    "notes": "add mask, add diff"
    }
)
train_dataset = dataset.GCDataset('Archive/train', 512, diff=True)
test_dataset = dataset.GCDataset('Archive/test', 512, diff=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = TransformerModel2(d_model=1024).train().cuda()
opt = Adam(model.parameters(), 1e-4)
criterion = CrossEntropyLoss()
nepochs = 100

for epoch in range(nepochs):
    train(epoch, model, train_dataloader, criterion, opt)

    test(epoch, model, test_dataloader, criterion)

    torch.save(model.state_dict(), f"ckpt/model{epoch}.pth")

    