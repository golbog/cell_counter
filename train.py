import argparse
import random
import time
from pathlib import Path

import torch
from torch import optim

from data_handler import dataset
from data_handler.augmentor import AlbumentationAugmentor, Cropper
from model.cell_counter import  PytorchCellCounter
from model.fpn import FPN
from model.val import Val
from model.resunetpp import ResUNetPP
from model.c_resunet import CResUnet
from model.train import Train
from model.losses import WeightedBceLoss, CombinedLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Cell Counter')
    parser.add_argument('--arch', type=str, default='FPN', choices=['FPN', 'ResUNetPP', 'CResUnet'], help='Model architecture')
    parser.add_argument('--train_images', type=str, default='data/train/images_original', help='Path to training images')
    parser.add_argument('--train_masks', type=str, default='data/train/masks_original', help='Path to training masks')
    parser.add_argument('--train_weights', type=str, default='data/train/weights', help='Path to training weights')
    parser.add_argument('--val_images', type=str, default='data/eval/images', help='Path to validation images')
    parser.add_argument('--val_masks', type=str, default='data/eval/masks', help='Path to validation masks')
    parser.add_argument('--save_name', type=str, default='best_model', help='Name for saved model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    if args.arch == 'FPN':
        net = FPN(1).float().to(device)
    elif args.arch == 'ResUNetPP':
        net = ResUNetPP(1).float().to(device)
    elif args.arch == 'CResUnet':
        net = CResUnet(1).float().to(device)

    criterion_train = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    criterion_val = WeightedBceLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.004)

    imgs_root = Path(args.train_images)
    masks_root = Path(args.train_masks)
    weights_root = Path(args.train_weights)

    data = dataset.CCDataset.load_dataset_structure(imgs_root, masks_root, weights_root)
    all_names = list(data.keys())
    random.shuffle(all_names)
    
    split_ratio = 0.8
    split_idx = int(len(all_names) * split_ratio)
    train_names = all_names[:split_idx]
    val_names = all_names[split_idx:]
    
    print(f"Splitting data: {len(train_names)} train, {len(val_names)} val")

    train_dataset = dataset.CCDataset(True, imgs_root, masks_root, weights_root,
                                      AlbumentationAugmentor(crop_size=(1024, 1024)), subset_names=train_names, data=data)
    
    val_dataset = dataset.CCDataset(True, imgs_root, masks_root, weights_root,
                                      Cropper(crop_size=(1024, 1024)), subset_names=val_names, data=data)
    
    test_dataset = dataset.CCDataset(False, args.val_images, args.val_masks, transform=Cropper(crop_size=(1024, 1024)))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True)
    
    trainer = Train(net, criterion_train, optimizer, train_loader, device=device)
    evaler = Val(net, criterion_val, eval_loader, device=device)
    tester = Val(net, criterion_val, test_loader, device=device)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for n in range(args.epochs):
        trainer.run_n_epochs(2)
        val_loss = evaler.eval(trainer.get_epoch_n())
        print(f"Validation loss after epoch {trainer.get_epoch_n()}: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            cc = PytorchCellCounter(net)
            cc.save(f'model/trained/{args.save_name}_early_stopping.pth')
            print(f"New best model saved with loss {best_val_loss}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} checks")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("Running evaluation on test set...")
    checkpoint = torch.load(f'model/trained/{args.save_name}_early_stopping.pth')
    if 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
        
    test_loss = tester.eval(0)
    print(f"Test loss: {test_loss}")

    cc = PytorchCellCounter(net)
    cc.save(f'{args.save_name}_{time.time()}.pth')
