import torch


class Val:
    def __init__(self, model, loss, data_loader, device='cuda'):
        self.model = model
        self.loss = loss
        self.data_loader = data_loader
        self.device = device

        self.completed_epochs = 0
        self.val_losses = dict()

    def eval(self, at_epoch):
        self.model.eval()
        val_loss = 0
        for i, data in enumerate(self.data_loader):
            images, gt, weights = data
            with torch.no_grad():
                images = images.to(self.device)
                gt = gt.to(self.device)
                weights = weights.to(self.device)

                outputs = self.model(images.float())
                val_loss = val_loss + self.loss(outputs, gt, weights).item()
        self.val_losses[at_epoch] = val_loss / len(self.data_loader)
        print(f'Test loss at epoch {at_epoch}: {self.val_losses[at_epoch]}')
        self.model.train()
        return self.val_losses[at_epoch]