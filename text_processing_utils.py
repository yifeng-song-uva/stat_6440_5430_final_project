import re
from collections import Counter
import numpy as np

def remove_punctuation_regex(text):
    text = text.replace("n't", " not") # deal with contractions
    text = re.sub("\'m|\'ll|\'s|\'d", " ", text)
    text = text.replace("'", " ")
    return re.sub(r'[^\w\s\n\t]', '', text)

def tokenize(text):
    text_vec = [e.strip().lower() for e in text.split(" ")]
    text_vec = [e for e in text_vec if e != ""]
    return text_vec

def create_bag_of_words(corpus, df_min, df_max_proportion, tfidf_threshold_q):
    '''
    INPUT:
    corpus: a dictionary of dictionaries: each inner dictionary maps each word to its count in the document
    df_min: minimum # of document frequency to keep a word in vocabulary
    df_max_proportion: maximum document frequency (in proportions) to keep a word in vocabulary
    tfidf_threshold_q: the threshold tf-idf quantile to keep a word in a document
    OUTPUT:
    word_counts_by_document: the updated corpus after fitering by df and tf-idf
    vocab_new: list of all unique words in the corpus after filtering
    n: total number of words in the updated corpus
    '''
    word_counts_by_document = {j:Counter(tokenize(rv)) for j,rv in enumerate(corpus)}
    print(1)
    document_length = {j:len(rv) for j,rv in enumerate(corpus)}
    tf = {}
    for k,v in word_counts_by_document.items():
        tf[k] = {}
        for k2,v2 in v.items():
            tf[k][k2] = v2/document_length[k]
    print(2)
    df = {}
    for k,v in tf.items():
        for k2, v2 in v.items():
            if v2 > 0:
                try:
                    df[k2] += 1
                except KeyError:
                    df[k2] = 1
    print(3)
    vocab = set()
    for k,v in df.items():
        if v/len(corpus) <= df_max_proportion and v >= df_min:
            vocab.add(k)
    idf = {k:np.log(len(corpus)/v) for k,v in df.items() if k in vocab}
    tfidf = {}
    for k,v in tf.items():
        tfidf[k] = {}
        for k2,v2 in v.items():
            if k2 in vocab:
                tfidf[k][k2] = v2 * idf[k2]
    all_tfidf_vals = []
    for k,v in tfidf.items():
        for k2,v2 in v.items():
            all_tfidf_vals.append(v2)
    threshold_tfidf = np.quantile(all_tfidf_vals, tfidf_threshold_q)
    tfidf_new = {}
    for k,v in tfidf.items():
        tfidf_new[k] = {}
        for k2,v2 in v.items():
            if v2 >= threshold_tfidf:
                tfidf_new[k][k2] = v2
    word_counts_by_document_new = {}
    n = 0
    vocab_new = set()
    for k,v in tfidf_new.items():
        word_counts_by_document_new[k] = {}
        for k2,v2 in v.items():
            n += word_counts_by_document[k][k2]
            vocab_new.add(k2)
            word_counts_by_document_new[k][k2] = word_counts_by_document[k][k2]
    return word_counts_by_document_new, vocab_new, n