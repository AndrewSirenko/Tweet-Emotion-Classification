Andrew Sirenko

Emotion Classification on tweets using GLoVe word embeddings. 

Contains all training and testing functions, run with 

python emotion_classification.py --model dense

python emotion_classification.py --model extension1

python emotion_classification.py --model RNN

python emotion_classification.py --model extension2

My 1st extension was step-decay training scheduler, which is named extension_train_model.

My 2nd extension was a CNN located in models.py under ExperimentalNetwork