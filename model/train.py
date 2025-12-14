import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class Train:
    def __init__(self, model, loss, optimizer, data_loader, device='cuda', scheduler=None, use_amp=False, max_grad_norm=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.max_grad_norm = max_grad_norm

        self.completed_epochs = 0
        self.losses = list()

    def run_n_epochs(self, n):
        self.model.train()
        for epoch in range(n):
            epoch_loss = 0
            with tqdm(self.data_loader, unit="batch", desc=f"Epoch {self.completed_epochs + 1}") as tepoch:
                for data in tepoch:
                    images, gt, weights = data
                    images = images.to(self.device)
                    gt = gt.to(self.device)
                    weights = weights.to(self.device)
                    self.optimizer.zero_grad()

                    if self.use_amp:
                        with autocast():
                            outputs = self.model(images.float())
                            loss = self.loss(outputs, gt, weights)
                        
                        self.scaler.scale(loss).backward()
                        if self.max_grad_norm:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        outputs = self.model(images.float())
                        loss = self.loss(outputs, gt, weights)
                        loss.backward()
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
            
            if self.scheduler:
                self.scheduler.step()

            self.completed_epochs += 1
            avg_loss = epoch_loss / len(self.data_loader)
            self.losses.append(avg_loss)
            print(f'Train loss for epoch {self.completed_epochs}: {avg_loss:.6f}')

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        self.model.eval()

    def get_epoch_n(self):
        return self.completed_epochs
