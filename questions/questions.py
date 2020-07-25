import nltk
import sys
import os
import string
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 2


def main():

    # Check command-line argumentsclrsc

    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for file in os.listdir(directory):
        f = open(os.path.join(directory, file), encoding="utf8")
        contents = f.read()
        files[file] = contents

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    lst = nltk.word_tokenize(document)
    tokens = []
    for word in lst:
        if word not in string.punctuation:
            if word.lower() not in nltk.corpus.stopwords.words("english"):
                tokens.append(word.lower())
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_of_doc = len(documents.keys())
    count = dict()
    for i in documents.keys():
        for word in set(documents[i]):
            count[word] = count.get(word, 0) + 1

    for i in count.keys():
        count[i] = math.log(num_of_doc/count[i])

    return count


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_rank = dict()
    for file in files.keys():
        count = Counter(files[file])
        for word in query:
            file_rank[file] = file_rank.get(file, 0) + (count[word]*idfs[word])
    file_rank = sorted(file_rank, key=lambda d: -file_rank[d])

    return file_rank[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sen_rank = dict()
    for s in sentences.keys():
        num = 0
        sen_rank[s] = {"idf": 0, "qtd": 0.0}
        count = Counter(sentences[s])
        for word in query:
            if word in sentences[s]:
                sen_rank[s]["idf"] += idfs[word]
                num += 1
        sen_rank[s]["qtd"] = num / len(sentences[s])

    def keyfunc(k):
        return -sen_rank[k]["idf"], -sen_rank[k]["qtd"]

    return sorted(sen_rank, key=keyfunc)[:n]


if __name__ == "__main__":
    main()
