from scipy.stats import pearsonr
from keras.preprocessing.text import Tokenizer

from utils import load_data_frame, preprocess_sentences, create_model


def run_sts():
    train_df = load_data_frame('dataset/train.txt')
    test_df = load_data_frame('dataset/test.txt')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_df['sentence_A']) + list(train_df['sentence_B']))

    train_seq_A, train_seq_B = preprocess_sentences(train_df, tokenizer)
    test_seq_A, test_seq_B = preprocess_sentences(test_df, tokenizer)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 4
    model = create_model(vocab_size, embedding_dim)
    model.summary()

    normalized_scores = train_df['relatedness_score'].values / 5.0
    model.fit([train_seq_A, train_seq_B], normalized_scores, epochs=10, batch_size=128)

    predicted_scores = model.predict([test_seq_A, test_seq_B]) * 5.0

    actual_scores = test_df['relatedness_score'].values
    correlation, _ = pearsonr(predicted_scores.flatten(), actual_scores)
    print(f'Pearson correlation: {correlation:.4f}')


if __name__ == '__main__':
    run_sts()
