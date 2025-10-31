import os
import nltk as nltk
from gensim.models import Word2Vec
import pandas as pd
import logging  # Setting up the loggings to monitor gensim
import warnings


warnings.filterwarnings('ignore')
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


path = r".your_AST_path.csv"
data = pd.read_csv(path, encoding='utf-8')
text = data['node']
sent = [row for row in text]
vector_size=128         #128,64
all_words = [nltk.word_tokenize(str(sntncs)) for sntncs in sent if sntncs]
corpus_file = r'.your_corpus_file.txt'

os.makedirs(os.path.dirname(corpus_file), exist_ok=True)

with open(corpus_file, 'w', encoding='utf-8') as f:
    for sentence_tokens in all_words:

        sentence_str = ' '.join(sentence_tokens)

        f.write(sentence_str + '\n')

model = Word2Vec(all_words, workers=4, min_count=1,vector_size=vector_size)
# Save word embedding model
model_file = r'ypur_word2vec_embedding.txt'
directory = os.path.dirname(model_file)

if not os.path.exists(directory):
    os.makedirs(directory)
model.wv.save_word2vec_format(model_file, binary=False)
# print("model saved")