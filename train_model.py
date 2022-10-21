"""
This script contains a function to train an instance of our FFN
"""


import torch
import numpy as np 
from initiate_model import initiate_model


# define training function
def train_model(train_loader, test_loader, batch_normalize = False, dropout = 0.0, epochs = 10, width = 32, depth = 1, learning_rate = 0.001):
    
    # set number of epochs
    epochs = epochs
    print(f"Going to train {epochs} epochs.")

    # initiate model instane, lossfunction, and optimizer
    model_instance, loss_function, optimizer = initiate_model(dropout = dropout,
                                                              width = width,
                                                              depth = depth,
                                                              learning_rate = learning_rate,
                                                              batch_normalize = batch_normalize)

    # initialize losses
    losses = torch.zeros(epochs)
    train_accuracy  = []
    test_accuracy   = []

    for epoch in range(epochs):

        if epoch == 0:
            print(f"Training epoch {epoch}")
        elif epoch % 10 == 0 and epoch + 1 != epochs:
            print(f"Training epoch {epoch},\n \
            last trainloss was {losses[epoch-1]},\n \
            last trainaccuracy was {train_accuracy[epoch-1]},\n \
            last testaccuracy was {test_accuracy[epoch-1]}")

        # switch on training mode
        model_instance.train()
        
        # loop over training data batches
        batch_accuracy  = []
        batch_loss      = []
        
        for X,y in train_loader:

            # forward pass and loss
            y_hat = model_instance(X)
            loss = loss_function(y_hat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batch_loss.append(loss.item())

            # compute accuracy
            pred = (torch.sigmoid(y_hat) > 0.5).float()
            batch_accuracy.append(100*torch.mean((pred == y).float()))
            # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        train_accuracy.append(np.mean(batch_accuracy))

        # and get average losses across the batches
        losses[epoch] = np.mean(batch_loss)

        # test accuracy
        model_instance.eval()
        X,y = next(iter(test_loader)) 
        with torch.no_grad():
            y_hat = model_instance(X)
        
        # compute accuracy
        pred = (torch.sigmoid(y_hat) > 0.5).float()
        test_accuracy.append(100*torch.mean((pred == y).float())) 
        
        # print final loss when training is done
        if epoch + 1 == epochs:
            print(f"Training finished after {epochs} epochs, \
            last mean loss was {losses[epoch]}")

    # end epochs
    
    # function output
    return train_accuracy, test_accuracy, losses, model_instance