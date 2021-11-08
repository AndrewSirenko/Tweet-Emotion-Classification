

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        super(DenseNetwork, self).__init__()

        
        # Create any layers and attributes your network needs.
        vocab_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        # create and load embedding layer with pretrained Glove embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.from_pretrained(embeddings)

        # for layer 1
        self.fc1 = nn.Linear(100, 36)

        # non lin fcn f
        self.relu = nn.ReLU()

        # for making y \in R^4
        self.fc2 = nn.Linear(36, 4)

        

    def forward(self, x):
        
        #embedding layer
        x = self.embedding(x)
        
        # sum Pooling 
        # reference: https://discuss.pytorch.org/t/how-to-perform-sum-pooling/3357/2
        x = torch.sum(x, dim=1)
        

        # linear fcn on pooled
        x = self.fc1(x)

        # nonlinear fcn between hidden layer 1 and 2
        x = self.relu(x)
        # last layer --> output of 4 labels --> softmax -->  output
        x = self.fc2(x)
    
        # We DON"T Do softmax here because cross entropy loss doesn't do that 
        return x



class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, num_layers):
        super(RecurrentNetwork, self).__init__()

        
        vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.num_layers = num_layers

        # create and load embedding layer with pretrained Glove embeddings
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.from_pretrained(embeddings)

        # Design GRU
        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, self.num_layers, batch_first=True)

        # activation fcn
        self.fcn1 = nn.Linear(self.embedding_dim, 4)

    # Gets length of sentence vector (somehow faster than torch.count_nonzero AND performs better! TA said was fine)
    @staticmethod
    def get_lengths(self, x):
        x_lengths = []
        for sentence in x:
            counter = 0
            for word in sentence:
                if word != 0:
                    counter += 1
            x_lengths.append(counter)
        return x_lengths
    

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        


        # Get real sentence lengths so padded items won't screw up our GRU
        x_lengths = torch.count_nonzero(x, dim=1)
    
        # Embedding layer
        embeds = self.embedding(x)
        
        # Pack the padded sequence
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embeds,x_lengths, enforce_sorted=False, batch_first=True)

        # Go through 2 layer GRU
        # ACCORDING TO TA MUST USE HIDDEN FOR OUTPUT
        __, hidden = self.gru(packed_input)

        # Take last hidden layer transform and  for output of 4 vals, which will be softmaxed in loss fcn
        output = self.fcn1(hidden[-1])
        
        return output



# references: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#    and "Text classification using Convolutional Neural Networks" on youtube
class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings):
        super(ExperimentalNetwork, self).__init__()
        
        vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
    


        # create and load embedding layer with pretrained Glove embeddings
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.from_pretrained(embeddings)


        #dropout to prevent overfitting!
        self.dropout = nn.Dropout(0.25)

        # Convolutional pooling for 1d layer 
        # 91 words per sentence vector in input  channel, 8 output channels, as windows of 3 words
        self.conv1 = nn.Conv1d(91, 8, 4)
        # another conv layer
        self.conv2 = nn.Conv1d(8, 16, 4)
        self.relu = nn.ReLU()

        # max pool the maxes 
        self.pool = nn.MaxPool1d(3)

        # dense layer 1
        self.fc1 = nn.Linear(496, 128)

        # dense layer 2
        self.fc2 = nn.Linear(128, 32)

        # convert to labels that are softmaxable
        self.fc3 = nn.Linear(32, 4)
        
        

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        
        # PADDING if sentence x isn't [128,91] (Than you Antonio)
        sent_length = x.shape[1]
        if sent_length != 91:
            pad_length = 91 - sent_length
            pad = torch.nn.ConstantPad2d((0,pad_length,0,0),0)
            x = pad(x)

            

            #embedding layer
        x = self.embedding(x)
        
        #2 conv layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # sum Pooling 
        # reference: https://discuss.pytorch.org/t/how-to-perform-sum-pooling/3357/2
        x = self.pool(x)
        
        # flatten
        x = x.view(-1, self.num_flat_features(x))

        # dense layer with relu activation
        x = self.fc1(x)
        x = self.relu(x)

        # Dropout to avoid overfit
        x= self.dropout(x)

        # dense layer with relu activation
        x = self.fc2(x)
        x = self.relu(x)

        # Dropout to avoid overfit
        x= self.dropout(x)

        # last layer --> output of 4 labels -->  output --> softmax in loss
        x = self.fc3(x)
    
        return x

    # helper function for flattening
    # src: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
