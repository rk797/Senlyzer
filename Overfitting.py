import torch


# Early stopping to reduce ocerfitting (goes well with any optimizer)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when the validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')
early_stopping = EarlyStopping(patience=3, verbose=True)
early_stopping(val_loss, model)



# Increase the Dropout probability (according to stackoverflow anything aove 0.5 is not effective)
self.drop = nn.Dropout(p=0.1) # <-- INCREASE THIS VALUE
# try values - 0.1,0.2,0.3,0.4,0.5
# possble to have positive effect - above 0.5


# use L1/L2 regularization
## L2(Ridge Regression) is preferred. (I can't find examples or code that can be used as guidance)

# Add noise to input while training 