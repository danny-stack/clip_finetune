class EarlyStopping:
   def __init__(self, patience=5, min_delta=0, verbose=True):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_loss = None
       self.early_stop = False
       self.verbose = verbose

   def __call__(self, val_loss):
       if self.best_loss is None:
           self.best_loss = val_loss
       elif val_loss > self.best_loss - self.min_delta:
           self.counter += 1
           if self.verbose:
               print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           self.best_loss = val_loss
           self.counter = 0