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

def get_sentence_thematic(vocab, sentence):
  model_path = get_model_path()
  model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")
  liste_res = []
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
      liste_res.append([st.mean(liste_dists), thematic])
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
            this_sims.setdefault(token, sim)
            #update mean
            this_sims[token] = st.mean([this_sims[token], sim])
      selected_tokens = [x[1]  for x in sorted([[s, w] for w, s in this_sims.items()], reverse=True)][:min_nb]
      dic[key] = selected_tokens
  #print(json.dumps(dic, indent = 2, ensure_ascii=False))
  return dic

if __name__=="__main__":
    import json
    voc_dir = "vocabularies"
    base_name = "test_voc_fr.json"
    base_voc_path = f"{voc_dir}/{base_name}"
    with open(base_voc_path) as f:
        liste = json.load(f)
    extended_voc = extend_vocabulary(liste)
    with open(f"{voc_dir}/extended_{base_name}", "w") as w:
        w.write(json.dumps(extended_voc, indent =2))

    for sentence in ["J'aime le sport", "En sport moi c'est le foot", "Je kiffe Jésus et les églises", "toto titi"]:
      print(sentence)
      print(get_sentence_thematic(extended_voc, sentence))
