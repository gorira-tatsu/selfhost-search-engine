from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import math

def text_to_token(text: str):
    return text.split()

def token_to_ngram(tokens: list[str], n: int = 3) -> list[list[str]]:
    n_gramized_token = [] 

    for token in tokens:
      token_length = len(token)

      n_gram_tokens = []
      token_start = 0
      while token_start + n <= token_length:
        n_gram_tokens.append(token[token_start:token_start+n])
        token_start = token_start + 1

      n_gramized_token.append(n_gram_tokens)

    return n_gramized_token

def delete_stopwords(words: list[str]) -> list[str]:
    with open("../data/NLTK_stopwords_list") as f:
      stopwords = f.read().split()

    for i, word in enumerate(words):
      for stopword in stopwords:
        if word == stopword:
          del words[i]

    return words

class InvertedIndex(BaseModel):
    index: Dict[str, List[int]]

def create_inverted_index(tokens: list[str]):
    tmp = {}
    for i, token in enumerate(tokens):
      if tmp.get(token) == None:
        tmp[token] = [i]
      else:
        tmp[token].append(i)
    return InvertedIndex(index=tmp)

class DocumentInverted(BaseModel):
    """In-memory document-level inverted index with postings and file offsets."""

    index: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)
    doc_positions: Dict[int, Dict[str, int]] = Field(default_factory=dict)
    total_length: int = 0

    class Config:
        arbitrary_types_allowed = True

    def add_chunk(self, doc_id: int, tokens: List[str], start: Optional[int] = None, end: Optional[int] = None):
        for position, token in enumerate(tokens):
            postings = self.index.setdefault(token, [])
            postings.append((doc_id, position))
        self.doc_positions[doc_id] = {"start": start, "end": end}
        self.total_length += len(tokens)

    @property
    def document_count(self) -> int:
        return len(self.doc_positions)

    @property
    def average_length(self) -> float:
        count = self.document_count
        return self.total_length / count if count else 0.0

    def has_term(self, term: str) -> bool:
        return term in self.index

    def term_postings(self, term: str) -> List[Tuple[int, int]]:
        return self.index.get(term, [])

def addDocumentInvertedIndex(docInvertedIndex: DocumentInverted, chunk: list[str], docId: int) -> DocumentInverted:
    chunkSentence = " ".join(chunk)
    chunkTokens = text_to_token(chunkSentence)
    docInvertedIndex.add_chunk(docId, chunkTokens)
    return docInvertedIndex


def create_document_inverted_index(document: dict[int, list[str]]) -> DocumentInverted:
    doc_index = DocumentInverted()
    for key, tokens in document.items():
        doc_index.add_chunk(key, tokens)
    return doc_index


def binarySearch(array: list, target: int, offset: int = 0) -> int:
    middleIndex = len(array) // 2
    if array[middleIndex] == target:
      return middleIndex+offset
    elif array[middleIndex] > target:
      return binarySearch(array[:middleIndex], target, offset)
    elif array[middleIndex] < target:
      return binarySearch(array[middleIndex+1:], target, offset + middleIndex + 1)


def nextPhrase(term: str, PreviousPosition: int, inverted_index: InvertedIndex):
    phraseIndexList = inverted_index[term]

    nextPositionIndex = binarySearch(phraseIndexList, PreviousPosition) + 1
    print(nextPositionIndex) 
    if len(phraseIndexList) > nextPositionIndex:
      return phraseIndexList[nextPositionIndex]
    return None


def calculate_tf_idf(docInverted: DocumentInverted, targetDocId: int, targetTerm: str):
    postings = docInverted.term_postings(targetTerm)
    targetTermDoc = [p for p in postings if p[0] == targetDocId]
    containTermDocument = {doc_id for doc_id, _ in postings}

    if not targetTermDoc:
        tf = 0.0
    else:
        tf = math.log2(len(targetTermDoc)) + 1

    documentCount = docInverted.document_count
    containTermDocumentCount = len(containTermDocument) or 1

    if documentCount == 0:
        idf = 0.0
    else:
        idf = math.log2(documentCount / containTermDocumentCount)

    return tf * idf


def rankBM25_DocumentAtATime(docInverted: DocumentInverted, docs: Dict[int, List[str]], targetTerm: str, k_1: float = 1.2, b: float = 0.75):
    targetTermsPositionList = docInverted.term_postings(targetTerm)
    if not targetTermsPositionList:
        return {}

    targetDoc: Dict[int, List[int]] = {}
    for doc_id, position in targetTermsPositionList:
        positions = targetDoc.setdefault(doc_id, [])
        positions.append(position)

    l_avg = docInverted.average_length or 1.0
    N = docInverted.document_count or len(docs)
    forDocScore: Dict[int, float] = {}
    for doc_id, positions in targetDoc.items():
        f_td = len(positions)
        l_d = len(docs.get(doc_id, [])) or 1
        n_t = len(targetDoc)

        TF_BM25 = (f_td * (k_1 + 1)) / ((f_td + k_1) * ((1 - b) + b * (l_d / l_avg)))
        idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)

        Score_BM25 = idf * TF_BM25
        forDocScore[doc_id] = Score_BM25

    return forDocScore

def searchWord(target: str, docInverted: DocumentInverted):
    containTargetList = docInverted.term_postings(target)
    if not containTargetList:
        return None

    docTargetCount: Dict[int, int] = {}
    for doc_id, _ in containTargetList:
        docTargetCount[doc_id] = docTargetCount.get(doc_id, 0) + 1

    most_id = max(docTargetCount, key=docTargetCount.get)
    return {"id": most_id, "count": docTargetCount[most_id]}

def FileToIndex(filePath: str):
    document_index = DocumentInverted()

    def _finalize_chunk(end_position: int, chunk: List[str], chunk_start: int, chunk_id: int):
        if not chunk:
            return chunk_id

        chunk_id += 1
        chunk_tokens = text_to_token(" ".join(chunk))
        document_index.add_chunk(chunk_id, chunk_tokens, chunk_start, end_position)

        if chunk_id % 1000 == 0:
            print(chunk_id)

        return chunk_id

    with open(filePath, encoding="utf-8") as f:
        chunk: List[str] = []
        chunk_id = 0
        chunk_start_position = 0

        while True:
            line_start_position = f.tell()
            line = f.readline()

            if not line:
                chunk_id = _finalize_chunk(line_start_position, chunk, chunk_start_position, chunk_id)
                break

            if line.strip():
                chunk.append(line.rstrip("\n"))
                continue

            chunk_id = _finalize_chunk(line_start_position, chunk, chunk_start_position, chunk_id)
            chunk.clear()
            chunk_start_position = f.tell()

    return document_index

def getDocumentByLine(start: int, end: int, TEXT_PATH: str):
    lines = [] 
    with open(TEXT_PATH, "r") as f:
        f.seek(start)
        while f.tell() < end:
            lines.append(f.readline())
    return lines

if __name__ == "__main__":
    #TEXT_PATH = "../data/enwiki-latest-pages-articles-multistream.xml"
    TEXT_PATH = "../data/hamlet_TXT_FolgerShakespeare.txt"
    
    document_index = FileToIndex(TEXT_PATH)
    print(document_index)

    result = searchWord("To", document_index)
    if result:
        for_doc_position = document_index.doc_positions[result["id"]]
        print(result)
        print(getDocumentByLine(for_doc_position["start"], for_doc_position["end"], TEXT_PATH))

# with open(TEXT_PATH, encoding="utf-8") as f:
   #     text = f.read()
   # paragraphs = [p for p in text.split("\n\n") if p.strip()]

   # docs: Dict[int, List[str]] = {}
   # for i, p in enumerate(paragraphs):
   #     toks = delete_stopwords(text_to_token(p))
   #     docs[i] = toks

   # doc_idx = create_document_inverted_index(docs)
   # print(f"[Index] docs={len(docs)}, avg_len={doc_idx.metaData.documentLengthAverage:.2f}")

   # def bm25_one(term: str, topk: int = 3):
   #     if term not in doc_idx.index:
   #         print(f'[1term] "{term}": no hit'); return
   #     scores = rankBM25_DocumentAtATime(doc_idx, docs, term, k_1=1.5, b=0.75)
   #     top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
   #     print(f'[1term] "{term}" top{topk}')
   #     for r, (d, s) in enumerate(top, 1):
   #         print(f"  #{r} doc={d} score={s:.4f}")

   # def bm25_multi(terms: List[str], topk: int = 3):
   #     acc = defaultdict(float)
   #     hit = False
   #     for t in terms:
   #         if t in doc_idx.index:
   #             hit = True
   #             for d, s in rankBM25_DocumentAtATime(doc_idx, docs, t, k_1=1.5, b=0.75).items():
   #                 acc[d] += s
   #     if not hit:
   #         print(f'[multi] {terms}: no hit'); return
   #     top = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:topk]
   #     print(f'[multi] {terms} top{topk}')
   #     for r, (d, s) in enumerate(top, 1):
   #         print(f"  #{r} doc={d} score={s:.4f}")

   # bm25_one("hamlet")
   # bm25_one("king")
   # bm25_multi(["hamlet", "king"])
