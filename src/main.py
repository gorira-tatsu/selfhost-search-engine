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

class DocumentInvertedIndex(BaseModel):
      index: Dict[str, List[Tuple[int, int]]]
      metaData: Dict[str, int]

def create_document_inverted_index(document: dict[list[str]]):
    tmp = {}
    documentAllLengthCount = 0
    for i, key in enumerate(document.keys()):
      tokens = document[key]
      documentAllLengthCount = documentAllLengthCount + len(tokens)
      for j, token in enumerate(tokens):
        if tmp.get(token) == None:
          tmp[token] = [tuple([key, j])]
        else:
          tmp[token].append(tuple([key, j]))
    
    return DocumentInvertedIndex(index=tmp, metaData={"documentCount": len(document.keys()), "documentLengthAverage": documentAllLengthCount / len(document.keys())})
    #return tmp

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

    documentCount = docInverted.metaData["documentCount"]
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
        l_avg = docInverted.metaData["documentLengthAverage"]

        N = docInverted.metaData["documentCount"]
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

if __name__ == "__main__":
    TEXT_PATH = "../data/hamlet_TXT_FolgerShakespeare.txt"

    with open(TEXT_PATH, encoding="utf-8") as f:
      shakespeare_doc = f.read()

    paragraphs = [p for p in shakespeare_doc.split("\n\n") if p.strip()]

    docs: Dict[int, List[str]] = {}
    for i, doc in enumerate(paragraphs):
      tokens = text_to_token(doc)
      tokens = delete_stopwords(tokens)
      docs[i] = tokens

    doc_indexes = create_document_inverted_index(docs)

    target = "I"
    result = searchWord(target, doc_indexes)

    if result != None:

        top_id = result["id"]
        count = result["count"]

        print(f'Top document for "{target}": id={top_id}, count={count}')
        preview = " ".join(docs[top_id][:100])
        print(preview)

