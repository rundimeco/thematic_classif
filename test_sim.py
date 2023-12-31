import re, json
from gensim.models import KeyedVectors
import statistics as st
import os

def get_model_path():
  path = "models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
  if os.path.exists(path) == True:
    return path
  os.makedirs("models", exist_ok=True)
  url = "https://www.thedreamviewer.com/box/data/dbs/1/models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
  os.system(f"wget {url} -O {path}")
  return path

def classify_sentences(sentence_list):
  model_path = get_model_path()
  model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")
  vocab = get_extended_voc(language = "fr")
  list_thematics = []
  for sentence in sentence_list:
    list_thematics.append(get_sentence_thematic(vocab, model, sentence))
  return list_thematics

def get_sentence_thematic(vocab, model, sentence):
  liste_res = []
  #TODO: catch cases when no word of input sentence is found
  for thematic, word_list in vocab.items():
      #thematic_sentence = " ".join(word_list)
      liste_dists = []
      for w1 in re.split(" |,", sentence):
        for w2 in word_list:
            try:
              dist = model.distance(w1, w2)
            except:
              continue
            liste_dists.append(dist) 
      val = liste_dists[0] 
      if len(liste_dists)>0:
        val = st.mean(liste_dists)
      liste_res.append([val, thematic])
  #print(sorted(liste_res)[:5])
  return [sorted(liste_res)[0][1]]

def extend_vocabulary(liste, min_sim = 0.5, min_nb = 20):
  model_path = get_model_path()
  model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")
  dic = {}
  for sub_thematic in liste:
      this_sims = {word:1 for word in sub_thematic}
      key = sub_thematic[0]
      for word in sub_thematic:
        try:
          sims = set(model.most_similar(word))
        except:
          continue
          # TODO: Handling mwu
          #tokens = re.split(" ", word)
          #if len(tokens)>1:
        for token, sim in sims:
          if sim>min_sim:
            this_sims.setdefault(token, [])
            #update mean
            this_sims[token].append(sim)
      this_sims = {token : st.mean(L) if len(L)>1 else L[0]}
      selected_tokens = [x[1]  for x in sorted([[s, w] for w, s in this_sims.items()], reverse=True)][:min_nb]
      dic[key] = selected_tokens
  #print(json.dumps(dic, indent = 2, ensure_ascii=False))
  return dic

def get_extended_voc(language = "fr", force = False):
  #TODO use langauge parameter
  voc_dir = "vocabularies"
  base_name = f"test_voc_{language}.json"
  base_voc_path = f"{voc_dir}/{base_name}"
  extended_voc_path = f"{voc_dir}/extended_{base_name}"
  if os.path.exists(extended_voc_path):
    print("Loading vocabulary...\n")
    with open(extended_voc_path) as f:
        extended_voc = json.load(f)
  else:
    print("Creating vocabulary...\n")
    with open(base_voc_path) as f:
      liste = json.load(f)
    extended_voc = extend_vocabulary(liste)
    with open(extended_voc_path, "w") as w:
      w.write(json.dumps(extended_voc, indent =2))
  return extended_voc
if __name__=="__main__":
  import json
  sentences = ["J'aime le sport", "En sport moi c'est le foot", "Je kiffe Jésus et les églises", "toto titi"]
  thematics = classify_sentences(sentences)
  for sent, theme in zip(sentences, thematics):
    print(sent)
    print("-->", theme)
