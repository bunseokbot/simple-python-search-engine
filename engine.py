import math
import string


class PreProcessor(object):
    STOPWORDS = [
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
        'wouldn', "wouldn't"
    ]

    def normalizer(self, row):
        return row.translate(str.maketrans('', '', string.punctuation)).lower()

    def tokenizer(self, row):
        return row.split()

    def ngram_tokenizer(self, row, n=4):
        return self.ngram(row, n)

    def stopwords(self, tokens):
        return [token for token in tokens if token not in self.STOPWORDS]

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

    def ngram(self, text, n=4):
        size = len(text)
        if n > size:
            raise ValueError("n cannot be bigger than text size")

        return list([''.join(v) for v in zip(*[text[i:] for i in range(n)])])

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


class KoreanChosungSearchEngine(SearchEngine):
    CHOSUNG_START_LETTER = 4352
    JAMO_START_LETTER = 44032
    JAMO_END_LETTER = 55203
    JAMO_CYCLE = 588

    CHOSUNG = {
        0x1100: 'ㄱ',
        0x1101: 'ㄲ',
        0x1102: 'ㄴ',
        0x1103: 'ㄷ',
        0x1104: 'ㄸ',
        0x1105: 'ㄹ',
        0x1106: 'ㅁ',
        0x1107: 'ㅂ',
        0x1108: 'ㅃ',
        0x1109: 'ㅅ',
        0x110A: 'ㅆ',
        0x110B: 'ㅇ',
        0x110C: 'ㅈ',
        0x110D: 'ㅉ',
        0x110E: 'ㅊ',
        0x110F: 'ㅋ',
        0x1110: 'ㅌ',
        0x1111: 'ㅍ',
        0x1112: 'ㅎ',
    }

    def tokenizer(self, row):
        row = row.translate(str.maketrans('', '', '은는이가과을에의')).lower()
        return [self.extract_chosung(text) for text in row.split()]

    def extract_chosung(self, text):
        result = ""
        for value in text:
            result += self.CHOSUNG[int(
                (ord(value) - self.JAMO_START_LETTER) / self.JAMO_CYCLE + self.CHOSUNG_START_LETTER
            )]
        
        return result


def mock_english_search_engine_process(rows):
    engine = SearchEngine()
    
    for row in rows:
        engine.index(row)

    term = 'militia'
    documents = engine.search_term(term)
    for document_id in documents:
        document = engine.get_document_by_id(document_id)
        if document:
            print(document)
            print("TF-IDF Score:", engine.tfidf(document_id, term))
            print("BM25 Score:", engine.bm25(document_id, term))

    del engine


def mock_korean_chosung_search_engine_process(rows):
    engine = KoreanChosungSearchEngine()
    
    for row in rows:
        engine.index(row)
    
    term = 'ㄷㅎㅁㄱ'
    documents = engine.search_term(term)
    for document_id in documents:
        document = engine.get_document_by_id(document_id)
        if document:
            print(document)
            print("TF-IDF Score:", engine.tfidf(document_id, term))
            print("BM25 Score:", engine.bm25(document_id, term))

    del engine


if __name__ == "__main__":
    with open("example.txt") as f:
        rows = f.read().split('\n')

    print("Term: militia")
    mock_english_search_engine_process(rows)

    print("\n")

    with open("korean.txt", encoding='utf-8') as f:
        rows = f.read().split('\n')

    print("Term: ㄷㅎㅁㄱ")
    mock_korean_chosung_search_engine_process(rows)
