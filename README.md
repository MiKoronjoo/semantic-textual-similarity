## Semantic Textual Similarity (STS) Model

---

**Objective**

The primary objective of this project was to measure the semantic similarity between pairs of sentences. This is an
essential task in NLP, underpinning systems like question answering, document summarization, information retrieval, and
more. The performance of the model is evaluated using the Pearson correlation between predicted similarity scores and
human-annotated scores.

---

**Dataset**

The dataset used in this project comprises English sentence pairs. The training set contains 4,500 pairs, while the test
set has 4,927 pairs. Each pair of sentences is associated with a relatedness score, which varies from 0 (indicating
completely different topics) to 5 (indicating full similarity).

Structure:

| pair_ID | sentence_A | sentence_B | relatedness_score |
| ------- | ---------- | ---------- |-------------------|

---

**Data Preprocessing**

1. **Tokenization**: The sentences were tokenized using Keras's Tokenizer utility. This utility converts each sentence
   into a sequence of integers, representing word indices in a dictionary.
2. **Padding**: Sequences will be padded to the length of the longest individual sequence, so that all sequences have a
   uniform length.

---

**Model Architecture**

1. **Inputs**: The model accepts two input sequences, each representing a sentence.
2. **Embedding**: An embedding layer converts the input sequences into dense vectors of fixed size. The embedding
   dimension chosen is 4 for better result, although typically larger dimensions (e.g., 50, 100) are used for more
   semantic richness.
3. **Shared LSTM**: A shared LSTM layer processes both embedded sequences. This shared approach ensures that both
   sentences are encoded in a similar semantic space.
4. **Concatenation**: The LSTM outputs for both sentences are concatenated to produce a unified representation.
5. **Dropout**: A dropout layer is added to reduce the risk of overfitting.
6. **Prediction**: A dense layer is used to predict the relatedness score of the sentence pair.

---

**Training**

1. **Loss Function**: Mean Squared Error (MSE) was used as the loss function, which measures the average squared
   difference between predicted and actual scores.
2. **Optimizer**: The Adam optimizer was used for model training.
3. **Epochs**: The model was trained for 10 epochs, with a batch size of 128.
4. **Target Normalization**: The relatedness scores were normalized to a 0-1 range for training by dividing them by 5.

---

**Evaluation**

The model's predictions on the test dataset were scaled back to the original 0-5 range by multiplying the outputs by 5.
The model's performance was then evaluated using the Pearson correlation between the predicted scores and the
human-annotated scores from the test set.

Result:

```
Pearson correlation: 0.2148
```

---

**Conclusions**

The model produced a Pearson correlation of approximately `0.21`.

In summary, while the implemented model offers a foundational approach to STS, optimizations in data preprocessing,
model architecture, and training can significantly boost its performance.

## How to Install

### Prerequisites:

1. Ensure you have Python installed on your machine. If not, download and install it
   from [python.org](https://www.python.org/).

2. It's recommended to use a virtual environment for your project to avoid potential package conflicts. You can
   use `virtualenv` or the built-in `venv` module in Python.

---

### Step-by-step Guide:

#### 1. Set up a Virtual Environment (optional but recommended):

1.1. Install `virtualenv` (if you haven’t done so):

```bash
pip install virtualenv
```

1.2. Navigate to project directory:

```bash
cd semantic-textual-similarity
```

1.3. Create a new virtual environment:

```bash
virtualenv .venv
```

or

```bash
python -m venv .venv
```

1.4. Activate the virtual environment:

- **Windows**:

```bash
.\.venv\Scripts\activate
```

- **Mac/Linux**:

```bash
source .venv/bin/activate
```

Your command prompt should change to show the name of the activated environment.

#### 2. Install the Requirements:

2.1. Install the packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install `pandas`, `keras` and `scipy`.

#### 3. Run the Code:

3.1. Run the script:

```bash
python main.py
```

---
