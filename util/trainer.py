import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import time
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, loaders, model, hyperparams, run_name=None):
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.model = model
        self.model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get Hyperparameters
        self.loss_fn = hyperparams.get('loss_fn', nn.CrossEntropyLoss())
        self.lr = hyperparams.get('lr', 0.001)
        self.weight_decay = hyperparams.get('weight_decay', 0.001)

        self.optimizer = hyperparams.get('optimizer', torch.optim.Adam)(
            model.parameters(), 
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.num_epochs = hyperparams.get('num_epochs', 100)
        
        # Choose Device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        # File Naming
        if run_name:
            self.model_save_dir = f'logs/{run_name}/model'
            log_dir = f'runs/{run_name}'
        else:
            datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.model_save_dir = 'logs/' + datetime_str + '/model'
            log_dir = None

        # Initialize Writer
        self.writer = SummaryWriter(log_dir=log_dir)

        self.best_valid_accuracy = 0
        self.best_train_accuracy = 0
        self.test_accuracy = None
        self.train_time = 0

        print("Device: ", self.device)
        print("Model: ", model)
        print("Trainable Parameters: ", print(self.model_param_count))

        return
    
    def train(self, num_epochs=None):
        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        self.model.to(self.device)
        start_time = time.time()

        # Training Loop
        for epoch in range(num_epochs):
            # Set the model to training mode
            self.model.train()
            train_loss, correct = 0, 0

            # Loop over the batches in the dataset
            for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target.float())
                train_loss += loss
                loss.backward()
                self.optimizer.step()
                pred = output.argmax(dim=1, keepdim=True)
                target = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            # Evaluate the model on the validation set
            train_loss /= len(self.train_loader.dataset)
            train_accuracy = 100. * correct/len(self.train_loader.dataset)
            self.writer.add_scalar("Train loss", train_loss, epoch)
            self.writer.add_scalar("Train accuracy", train_accuracy, epoch)
            self.best_train_accuracy = max(self.best_train_accuracy, train_accuracy)

            
            # VALIDATION EVALUATION
            self.evaluate("valid", epoch)
            self.writer.flush()
            self.model.save(epoch, self.optimizer, self.model_save_dir)

        self.train_time = time.time() - start_time


    def evaluate(self, mode="valid", epoch=0):
        '''
        Evaluate model on dataset.
        Arguments:
            mode -- either "valid" or "test"
            epoch -- only applicable if mode == "valid"
        '''
        loader = self.val_loader if mode == "valid" else self.test_loader

        self.model.eval()
        loss, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.loss_fn(output, target.float()).item()
                pred = output.argmax(dim=1, keepdim=True)
                target = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(loader.dataset)
        accuracy = 100. * correct/len(loader.dataset)

        if mode == "valid":
            self.writer.add_scalar(f"Valid loss", loss, epoch)
            self.writer.add_scalar(f"Valid accuracy", accuracy, epoch)
            self.best_valid_accuracy = max(self.best_valid_accuracy, accuracy)

        elif mode == "test":
            self.writer.add_scalar(f"Test loss", loss)
            print("Test loss", loss)
            self.writer.add_scalar(f"Test accuracy", accuracy)
            print("Test accuracy", accuracy)
            self.test_accuracy = accuracy

    def print_stats(self):
        print("\n------------------------------------------------")
        print(f"Best Training Accuracy: {round(self.best_train_accuracy, 2)}%")
        print(f"Best Validation Accuracy: {round(self.best_valid_accuracy, 2)}%")
        print(f"Test Accuracy: {round(self.test_accuracy, 2)}%")
        print(f"Total Training Time: {round(self.train_time, 2)}s")
        print("------------------------------------------------")