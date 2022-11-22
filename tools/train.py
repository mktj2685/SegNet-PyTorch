from models.segnet import SegNet
from datasets.voc2012 import VOC2012
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from PIL import Image
import numpy as np
import datetime
import argparse
import logging
logging.basicConfig(level=logging.INFO)

COLOR_PALLET = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0,
                128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int,
                        help='number of epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='how many samples per batch to load.')
    parser.add_argument('--init_lr', default=1e-2,
                        type=float, help='initial learning rate.')
    return parser.parse_args()


def train_1epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    numel = 0
    epoch_acc = 0.
    epoch_loss = 0.
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(
            device)       # target : (B, 1, H, W)
        # target : (B, H, W)
        target = torch.squeeze(target, dim=1)
        # output : (B, num_classes, H, W)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        probs = torch.softmax(output, dim=1)
        # preds : (B, H, W)
        preds = probs.argmax(dim=1)
        epoch_acc += torch.sum(preds == target).item()
        epoch_loss += loss.item()
        numel += torch.numel(target)
    epoch_acc = 100 * epoch_acc / numel
    epoch_loss = epoch_loss / len(dataloader)
    return epoch_acc, epoch_loss


def validate_1epoch(
    model,
    dataloader,
    criterion,
    device
):
    numel = 0
    epoch_acc = 0.
    epoch_loss = 0.
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(
                device)       # target : (B, 1, H, W)
            # target : (B, H, W)
            target = torch.squeeze(target, dim=1)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            epoch_acc += torch.sum(preds == target).item()
            epoch_loss += criterion(output, target).item()
            numel += torch.numel(target)
    epoch_acc = 100 * epoch_acc / numel
    epoch_loss = epoch_loss / len(dataloader)
    return epoch_acc, epoch_loss


def predict(imgs):
    output = model(imgs)
    probs = torch.softmax(output, dim=1)
    preds = probs.argmax(dim=1)
    return preds


if __name__ == '__main__':
    # fix seed
    torch.manual_seed(0)

    # parse arguments.
    args = parse_args()

    # check gpu availability.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create dataLoader.
    num_classes = 21
    train_loader = DataLoader(VOC2012(phase='train'),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VOC2012(phase='val'),
                            batch_size=args.batch_size, shuffle=False)

    # create model.
    model = SegNet(21)
    model.to(device)
    summary(model, (1, 3, 415, 415))

    # create loss function
    # see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr)

    # create scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # loop run epoch
    best_loss = float('inf')
    best_acc = -float('inf')
    writer = SummaryWriter('logs')
    for i in range(1, args.epoch+1):
        logging.info(f'[{datetime.datetime.now()}] start epoch : {i}')
        train_acc, train_loss = train_1epoch(
            model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = validate_1epoch(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)
        logging.info(
            f'''
            epoch:{i},
            train loss:{train_loss:.2f},
            train acc:{train_acc:.2f},
            val loss:{val_loss:.2f},
            val acc:{val_acc:.2f}
            '''
        )
        # save weights
        if best_acc < val_acc:
            torch.save(model.state_dict(), 'weights/best_acc.pth')
            best_acc = val_acc
        if best_loss > val_loss:
            torch.save(model.state_dict(), 'weights/best_loss.pth')
            best_loss = val_loss
        # write logs
        # Scalar
        writer.add_scalar('Acc/train', train_acc, i)
        writer.add_scalar('Acc/val', val_acc, i)
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/val', val_loss, i)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i)
        # Image
        img, gt = val_loader.dataset[0]                                                         # img : (3, H, W), gt : (1, H, W)
        preds = predict(img[None, :, :, :].to(device))                                          # preds : (1, 1, H, W)
        preds = Image.fromarray(preds.squeeze().to(torch.uint8).cpu().numpy()).convert('P')     # preds : (H, W)
        preds.putpalette(COLOR_PALLET)
        preds = preds.convert('RGB')
        preds = np.asarray(preds, dtype=np.uint8).transpose(2, 0, 1)                            # preds : (3, H, W)
        gt = Image.fromarray(gt.squeeze().to(torch.uint8).cpu().numpy()).convert('P')           # gt : (H, W)
        gt.putpalette(COLOR_PALLET)
        gt = gt.convert('RGB')
        gt = np.asarray(gt, dtype=np.uint8).transpose(2, 0, 1)                                  # gt : (3, H, W)
        writer.add_image('Image/original', img, i)
        writer.add_image('Image/pred', preds, i)
        writer.add_image('Image/gt', gt, i)

    writer.close()
