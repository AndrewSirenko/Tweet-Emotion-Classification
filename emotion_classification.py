

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.

# Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of the models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """

    # Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    # Loop through the whole train dataset performing batch optimization with optimizer 
    print("Training Data...\n")

    last_epoch_loss = -1
    for epoch in range(10):  
        
        #switch to train mode
        model.train()
        cur_epoch_dev_loss = 0.0
        
        #Train through training set 
        for i, data in enumerate(train_generator, 0):
            # Gets inputs and labels
            X_train, y_train = data

            # ZERO GRAD PARAMS
            optimizer.zero_grad()

            # Go forward, back, optimize
            y_predicted = model(X_train)
            loss = loss_fn(y_predicted, y_train)
            loss.backward()
            optimizer.step()

        # Test on dev set, switching to eval mode to not accumulate dropout or grad when testing
        model.eval()
        with torch.no_grad():
            for data in dev_generator:
                X_dev, y_dev = data
                # Predict on dev
                y_predicted = model(X_dev)
                loss = loss_fn(y_predicted, y_dev)

                # Add to running dev loss
                cur_epoch_dev_loss += loss.item()

        # print dev set loss each epoch to stdout
        print('epoch %d: total dev set loss: %.3f' % (epoch + 1, cur_epoch_dev_loss))

        # early stopping (first side of and checks if there has been last epoch loss)
        if (last_epoch_loss >= 0 and last_epoch_loss < cur_epoch_dev_loss):
            break

        last_epoch_loss = cur_epoch_dev_loss


    return model

# EXTENSION 1 "extension-grading" Step Decay Training Scheduler 
# features 2 new params (drop, step_size) (see big comment)
# reference: https://towardsdatascience.com/learning-rate-scheduler-d8a55747dd90
def extension_train_model(model, loss_fn, optimizer, train_generator, dev_generator, drop, step_size):
    """
    Perform the training of the model based on the train and dev sets.
    WITH step-decay learning rate training scheduler 

    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set

    *NEW* :param drop: how much learning rate reduces every epoch (aka gamma)
    *NEW* :param step_size - num epochs till learning rate reduces
    :return model, the trained model
    """

    # Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    # Loop through the whole train dataset performing batch optimization with optimizer 
    print("Training Data...\n")

    
    # EXTENSION 1 CHANGED
    # this is the scheduler we step with in order to dec learning rate each epoch
    # computes dropped learning rate based on what epoch we're at
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=drop)

    # tracks learning rates used
    learning_rates = []

    last_epoch_loss = -1
    for epoch in range(10):  
        


        model.train()
        cur_epoch_dev_loss = 0.0
        
        #Train through training set 
        for i, data in enumerate(train_generator, 0):
            # Gets inputs and labels
            X_train, y_train = data

            # ZERO GRAD PARAMS
            optimizer.zero_grad()

            # Go forward, back, optimize
            y_predicted = model(X_train)
            loss = loss_fn(y_predicted, y_train)
            loss.backward()
            optimizer.step()




        # Test on dev set
        model.eval()
        with torch.no_grad():
            for data in dev_generator:
                X_dev, y_dev = data
                

                # Predict on dev
                y_predicted = model(X_dev)
                loss = loss_fn(y_predicted, y_dev)

                # Add to running dev loss
                cur_epoch_dev_loss += loss.item()
        
        # EXTENSION 1 CHANGED
        # tracks last learning rate (for debugging) then steps through scheduler 
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        # print dev set loss each epoch to stdout
        print('epoch %d: Lrate: %.5f: total dev set loss: %.3f' % (epoch + 1, learning_rates[-1], cur_epoch_dev_loss))

        # early stopping (first side of and checks if there has been last epoch loss)
        if (last_epoch_loss >= 0 and last_epoch_loss < cur_epoch_dev_loss):
            break

        last_epoch_loss = cur_epoch_dev_loss


    return model

def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()



    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result



    # Hyperparameter vars
    LRATE = 0.001
    RNN_HIDDEN_LAYERS = 2

    # For training scheduler
    DROP = 0.5
    STEP_SIZE = 6



    # Depending on user's choice of extensions, for each model we create, train, and test model
    choice = args.model

    '''CREATE MODEL'''
    # dense model see models.py
    if choice == "dense":
        model = models.DenseNetwork(embeddings)

    # GRU RNN model see models.py
    elif choice == "RNN":
        model = models.RecurrentNetwork(embeddings, RNN_HIDDEN_LAYERS)

    # extension 2 is the experimental model
    elif choice == "extension2":  
        model = models.ExperimentalNetwork(embeddings)
    # Default is Dense model for training scheduler
    else: 
        model = models.DenseNetwork(embeddings)


    '''TRAIN MODEL''' 
    # Choose optimizer (chose Adam because slides said it worked best!)
    optimizer = optim.Adam(model.parameters(), lr=LRATE)

    # Train model
    # extension 1 is the training scheduler
    if choice == "extension1":
        trained_model = extension_train_model(model, loss_fn, optimizer, train_generator, dev_generator, DROP, STEP_SIZE)
    # else use regular train_model
    else: 
        trained_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)


    '''TEST MODEL'''
    test_model(trained_model, loss_fn, test_generator)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
