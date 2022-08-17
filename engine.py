import math
import string


class PreProcessor(object):
    def normalizer(self, row):
        return row.translate(str.maketrans('', '', string.punctuation)).lower()

    def tokenizer(self, row):
        return row.split()

    def stopwords(self, tokens):
        return tokens

    def stemming(self, token):
        return token

    def lemmatisation(self, token):
        return token

    def indexing(self, tokens, document_id):
        for token in tokens:
            if token not in self._inverted_index_table:
                self._inverted_index_table[token] = [document_id]
            elif document_id not in self._inverted_index_table[token]:
                self._inverted_index_table[token].append(document_id)

            if token not in self._document_terms_frequency:
                self._document_terms_frequency[token] = {document_id: 1}
            elif document_id not in self._document_terms_frequency[token]:
                self._document_terms_frequency[token][document_id] = 1
            else:
                self._document_terms_frequency[token][document_id] += 1

    def get_index_table(self):
        return self._inverted_index_table

    def issue_document_id(self):
        self._last_document += 1
        return self._last_document

    def get_document_by_id(self, document_id):
        return self._document_table.get(document_id, None)

    def get_average_document_len(self):
        return sum([len(document) for document in self._document_table.values()]) / self._last_document


class SearchEngine(PreProcessor):
    def __init__(self):
        self._inverted_index_table = {}
        self._document_table = {}
        self._document_terms_frequency = {}
        self._last_document = 0

    def index(self, document):
        document_id = self.issue_document_id()

        # Pre-processing
        normalized_document = self.normalizer(document)
        tokens = self.tokenizer(normalized_document)
        tokens = self.stopwords(tokens)

        # Indexing
        self.indexing(tokens, document_id)

        self._document_table[document_id] = document

    def search_term(self, term):
        term = self.normalizer(term)
        if term not in self._inverted_index_table:
            return

        documents = self._inverted_index_table[term]
        if not documents:
            return

        return documents

    def search_sentence(self, sentence):
        normalized_sentence = self.normalizer(sentence)
        tokens = self.tokenizer(normalized_sentence)
        tokens = self.stopwords(tokens)

        documents = []
        for token in tokens:
            for document_id in self.search_term(token):
                if document_id not in documents:
                    documents.append(document_id)

        return sorted(documents)

    def tf(self, document_id, term):
        return self._document_terms_frequency[term][document_id]

    def df(self, term):
        if term not in self._inverted_index_table:
            return 0

        return len(self._inverted_index_table[term])

    def idf(self, term):
        return math.log(self._last_document / (1 + self.df(term)))

    def tfidf(self, document_id, term):
        term = self.normalizer(term)
        return self.tf(document_id, term) * self.idf(term)

    def bm25(self, document_id, term):
        # Parameters
        k = 1.2
        b = 0.75
        D = len(self.get_document_by_id(document_id))

        return self.idf(term) * (
            (self.tf(document_id, term) * (k + 1)) / (self.tf(document_id, term) + k * (1 - b + b * D / self.get_average_document_len()))
        )


def mock_search_engine_process(rows):
    engine = SearchEngine()
    
    for row in rows:
        engine.index(row)

    term = 'in'
    documents = engine.search_term(term)
    for document_id in documents:
        document = engine.get_document_by_id(document_id)
        if document:
            print(document, engine.tfidf(document_id, term))
            print(document, engine.bm25(document_id, term))

    del engine


if __name__ == "__main__":
    with open("example.txt") as f:
        rows = f.read().split('\n')

    mock_search_engine_process(rows)
