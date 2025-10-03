from pydantic import BaseModel
from typing import Dict, List

def text_to_token(text: str):
  return text.split()

#def text_to_ngram(text: list[str], n: int):


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

def nextPhrase(term: str, nowPosition: int, inverted_index: InvertedIndex):
  phraseIndex = inverted_index[term]
 
  phraseLength = len(phraseIndex) 
  phraseNowPosition = phraseIndex[phraseLength / 2] 
  while i <= phraseLength / 2:
    if phraseNowPosition < nowPosition
    phraseIndex[phraseLength / 2]
    i = i + 1
    

if __name__ == "__main__":
  with open("../data/tiny_hamlet.txt") as f:
    shakespere_test = f.read()

  print(" ".join(delete_stopwords(text_to_token(shakespere_test))))

  print("\n" + " ".join(shakespere_test.split()))

  print(create_inverted_index(delete_stopwords(text_to_token(shakespere_test))))

  #print(token_to_ngram(["Orienterring", "Smithnian"]))

