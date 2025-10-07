from pydantic import BaseModel
from typing import Dict, List, Tuple
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

class CorpusMeta(BaseModel):
    documentCount: int
    documentLengthAverage: float

class DocumentInvertedIndex(BaseModel):
    index: Dict[str, List[Tuple[int, int]]]
    metaData: CorpusMeta

def addDocumentInvertedIndex(docInvertedIndex: DocumentInvertedIndex, chunk: list[str], docId: int):
    tmp = {}
    chunkSentence = " ".join(chunk)
    chunkTokens = text_to_token(chunkSentence)    
    chunkDocumentLength = len(chunkTokens)
    
    for i, token in enumerate(chunkTokens):
        postings = docInvertedIndex.index.setdefault(token, [])
        postings.append((docId, i))
        docInvertedIndex.index[token] = sorted(postings)

    prev_count = docInvertedIndex.metaData.documentCount
    prev_avg = docInvertedIndex.metaData.documentLengthAverage
    new_count = prev_count + 1
    if new_count == 0:
        new_avg = 0.0
    else:
        # keep running mean to avoid recalculating from scratch
        new_avg = ((prev_avg * prev_count) + chunkDocumentLength) / new_count
    docInvertedIndex.metaData.documentCount = new_count
    docInvertedIndex.metaData.documentLengthAverage = new_avg

    return docInvertedIndex

def create_document_inverted_index(document: dict[int, list[str]]):
    tmp = {}
    documentAllLengthCount = 0
    for key, tokens in document.items():
        documentAllLengthCount += len(tokens)
        for j, token in enumerate(tokens):
            tmp.setdefault(token, []).append((key, j))

    doc_count = len(document)
    avg_len = documentAllLengthCount / doc_count if doc_count else 0.0

    return DocumentInvertedIndex(
        index=tmp,
        metaData=CorpusMeta(
            documentCount=doc_count,
            documentLengthAverage=avg_len,
        ),
    )


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


def calculate_tf_idf(docInverted, targetDocId: int, targetTerm: str):
    targetTermDoc = []
    containTermDocument = []
    for index in docInverted.index[targetTerm]:
      if index[0] == targetDocId: 
        targetTermDoc.append(index)
      containTermDocument.append(index[0])

    if len(targetTermDoc) == 0:
      tf = 0
    else:
      tf = math.log2(len(targetTermDoc)) + 1

    documentCount = docInverted.documentCount
    containTermDocumentCount = len(set(containTermDocument))

    idf = math.log2(documentCount/containTermDocumentCount)

    return tf * idf


def rankBM25_DocumentAtATime(docInverted, docs, targetTerm: str, k_1: float = 1.2, b: float = 0.75):
    targetTermsPositionList = docInverted.index[targetTerm]
    targetDoc = {}
    for index in targetTermsPositionList:
        if targetDoc.get(index[0]) == None:
            targetDoc[index[0]] = [index[1]]
        else:
            targetDoc[index[0]].append(index[1])

    forDocScore = {}
    for doc in targetDoc.keys():
        f_td = len(targetDoc[doc])
        l_d = len(docs[doc])
        l_avg = docInverted.metaData.documentLengthAverage

        N = docInverted.metaData.documentCount
        n_t = len(targetDoc.keys())

        TF_BM25 = (f_td * (k_1 + 1)) / ((f_td + k_1) * ((1 - b) + b * (l_d / l_avg)))
        idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)

        Score_BM25 = idf * TF_BM25
        forDocScore[doc] = Score_BM25

    return forDocScore

def searchWord(target: str, docInverted):
    try:
      containTargetList = docInverted.index[target]
    except KeyError:
      return None
    docTargetCount = {}
    for t in containTargetList:
      if docTargetCount.get(t[0]) == None:
        docTargetCount[t[0]] = 1
      else:
        docTargetCount[t[0]] = docTargetCount[t[0]] + 1

    mostList = {"id": 0, "count": 0}
    for docTargetCountKey in docTargetCount.keys():
      if docTargetCount[docTargetCountKey] > mostList["count"]:
        mostList["id"] = docTargetCountKey
        mostList["count"] = docTargetCount[docTargetCountKey]

    return mostList

def FileToIndex(filePath: str):
    docInvertedIndex = None
    docIdFilePosition: Dict[int, Dict[str, int]] = {}

    def _finalize_chunk(end_position: int, chunk: List[str], chunk_start: int, chunk_id: int):
        nonlocal docInvertedIndex
        if not chunk:
            return chunk_id

        chunk_id += 1
        chunk_tokens = text_to_token(" ".join(chunk))

        if docInvertedIndex is None:
            docInvertedIndex = create_document_inverted_index({chunk_id: chunk_tokens})
        else:
            docInvertedIndex = addDocumentInvertedIndex(docInvertedIndex, chunk_tokens, chunk_id)

        docIdFilePosition[chunk_id] = {
            "start": chunk_start,
            "end": end_position,
        }

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

    return docInvertedIndex, docIdFilePosition

def getDocumentByLine(start: int, end: int, TEXT_PATH: str):
    lines = [] 
    with open(TEXT_PATH, "r") as f:
        f.seek(start)
        while f.tell() < end:
            lines.append(f.readline())
    return lines

if __name__ == "__main__":
    from collections import defaultdict

    #TEXT_PATH = "../data/enwiki-latest-pages-articles-multistream.xml"
    TEXT_PATH = "../data/hamlet_TXT_FolgerShakespeare.txt"
    
    docInverted = FileToIndex(TEXT_PATH)
    print(docInverted)

    result = searchWord("To", docInverted[0])
    forDocPosition = docInverted[1][result["id"]]
    print(result)
    print(getDocumentByLine(forDocPosition["start"], forDocPosition["end"], TEXT_PATH))

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
