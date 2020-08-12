import time
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2,
                                  stride=2)
        self.l2 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2,
                                  stride=2)
        self.l3 = torch.nn.Conv1d(in_channels=16, out_channels=32,
                                  kernel_size=2, stride=2)
        self.l4 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=16,
                                           kernel_size=2, stride=2)
        self.l5 = torch.nn.ConvTranspose1d(in_channels=16, out_channels=8,
                                           kernel_size=2, stride=2)
        self.l6 = torch.nn.ConvTranspose1d(in_channels=8, out_channels=1,
                                           kernel_size=2, stride=2)

        self.actv1 = torch.nn.Tanh()
        self.actv2 = torch.nn.LeakyReLU()

        self.dropout = torch.nn.Dropout(0.2)
        self.out = torch.nn.Linear(1024, 1024)

    def forward(self, X):
        z = self.actv1(self.l1(X))
        z = self.dropout(z)
        z = self.actv1(self.l2(z))
        z = self.actv1(self.l3(z))
        z = self.actv1(self.l4(z))
        z = self.actv1(self.l5(z))
        z = self.dropout(z)
        z = self.actv1(self.l6(z))

        return self.out(z)


def train(model, device, optimizer, scheduler,
          train_dataloader, validation_dataloader, epochs, seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_fn = torch.nn.MSELoss()
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_loss = []
    val_loss = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        if epoch_i > 25:
            scheduler.step()
            print(scheduler.get_lr())
        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            x_batch = batch[0].to(device).unsqueeze(1)
            y_batch = batch[1].to(device).unsqueeze(1)

            model.zero_grad()

            y_pred = model(x_batch)

            # Compute and print loss.
            loss = loss_fn(y_pred, y_batch)
            if not step % 1000:
                print(step, torch.sqrt(loss).item())

            # Accumulate the training loss over all of the batch
            total_train_loss += loss.item()

            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_loss.append(avg_train_loss)

        training_time = time.time() - t0

        print("")
        print(f"Average training loss: {avg_train_loss}")
        print(f"Training epcoh took: {training_time}")

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_eval_MSE = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            x_batch = batch[0].to(device).unsqueeze(1)
            y_batch = batch[1].to(device).unsqueeze(1)

            with torch.no_grad():

                pred = model(x_batch)

                # Compute and print loss.
                loss = loss_fn(pred, y_batch)
            if not step % 100:
                print(step, torch.sqrt(loss).item())

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_MSE += loss.item()

        # Report the final accuracy for this validation run.
        avg_MSE_accuracy = total_eval_MSE / len(validation_dataloader)
        val_loss.append(avg_MSE_accuracy)
        print(f"  MSE: {avg_MSE_accuracy}")

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = time.time() - t0

        print(f"  Validation Loss: {avg_val_loss}")
        print(f"  Validation took: {validation_time}")

    print("")
    print("Training complete!")

    return val_loss, training_loss,model