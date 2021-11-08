
# Twitter emotion classification using deep learning

This repo uses pre-trained GloVe word embeddings and a variety of deep learning models to test emotion classification on tweets from the CrowdFlower dataset.


## Run Locally

Clone the project

```bash
  git clone https://github.com/AndrewSirenko/Tweet-Emotion-Classification.git
```

Go to the project directory

```bash
  cd my-project
```

Download 100 dimensional pretrained GloVe embeddings from https://nlp.stanford.edu/projects/glove/

```bash
  pip install
```

Train the models

```python
    python emotion_classification.py --model dense

    python emotion_classification.py --model extension1

    python emotion_classification.py --model RNN

    python emotion_classification.py --model extension2
```
My 1st extension was step-decay training scheduler, which is named extension_train_model.

My 2nd extension was a CNN located in models.py under ExperimentalNetwork
