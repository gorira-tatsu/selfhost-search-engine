from pydantic import BaseModel
from typing import Dict, List, Tuple
import math

def text_to_token(text: str):
  return text.split()

def token_to_ngram(tokens: list[str], n: int = 3):
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

def delete_stopwords(words: list[str]):
  with open("../data/NLTK_stopwords_list") as f:
    stopwords = f.read().split()

  for i, word in enumerate(words):
    for stopword in stopwords:
      if word == stopword:
        print(word, stopword, i) 
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
  for i, key in enumerate(document.keys()):
    tokens = document[key]
    for j, token in enumerate(tokens):
      if tmp.get(token) == None:
        tmp[token] = [tuple([key, j])]
      else:
        tmp[token].append(tuple([key, j]))
  return DocumentInvertedIndex(index=tmp, metaData={"documentCount": len(document.keys())})
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

if __name__ == "__main__":
  with open("../data/tiny_hamlet.txt") as f:
    shakespere_test = f.read()

  print(" ".join(delete_stopwords(text_to_token(shakespere_test))))

  print("\n" + " ".join(shakespere_test.split()))

  Indexes = create_inverted_index(delete_stopwords(text_to_token(shakespere_test)))
  print(Indexes)

  print(nextPhrase("To", 54, Indexes.index)) 

  print(token_to_ngram(["Orienterring", "Smithnian"]))

  with open("../data/hamlet_TXT_FolgerShakespeare.txt") as f:
    shakespeare_doc = f.read()
  
  docs = {}
  for i, doc in enumerate(shakespeare_doc.split("\n\n")):
    docs[i] = doc.split()
  
  #doc_Indexes = create_document_inverted_index(docs)

  document_tiny = {
    1: "Do you quarrel, sir?".split(),
    2: "Quarrel sir! no, sir!".split(),
    3: "If you do, sir, I am for you: I serve as good a man as you".split(),
    4: "No better".split(),
    5: "Well, sir".split()
    }

  docIndexes = create_document_inverted_index(document_tiny)
  print(docIndexes)
 
  document_tiny_tfidf = {}
  for docKey in document_tiny.keys():
    termsTFIDF = [] 
    for term in document_tiny[docKey]:
      termsTFIDF.append(calculate_tf_idf(docIndexes, docKey, "sir")) 
    document_tiny_tfidf[docKey] = termsTFIDF
  
  print(document_tiny_tfidf)
