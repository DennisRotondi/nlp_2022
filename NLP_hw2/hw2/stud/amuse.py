import torch
import copy
import csv
import requests
import json
from typing import Tuple, List, Any, Dict, Optional, Union

class AMuSE_WSD_online():
    def __init__(self, language: str, notebook: bool = True, filter_layer: bool = True) -> None:
        # this class allow to use the api of AMuSE_WSD that the professor and TA's adviced to use as EXTRA task

        # ok, there are many possible ways to solve the predicate disambiguation task, the most efficient are very
        # similar to the other 2 I've implemented, so I've decided to work on this extra to first try the simplest
        # thing possible: I let AMuSE-WSD do the work. Given the predicted sense is possible to encode this
        # information to use a neural network or any machine learning model properly fitted. 
        def read_in_dict(in_file: str):
            with open(in_file) as file:
                tsv_file = csv.reader(file, delimiter="\t")
                dict = {}
                for line in tsv_file:
                    key, val = line[0], line[1]
                    dict[key] = val
            return dict
        # we take care of relative imports
        if notebook:
            plus = "../../"
        else:
            plus = ""
        bn2va_file = plus+"model/amuse/VA_bn2va.tsv"
        va_info_file = plus+"model/amuse/VA_frame_info.tsv"
        # convert from bn synset to verbatlas frame
        self.bn2va = read_in_dict(bn2va_file)
        # retrive the sense from verbatlas frame
        self.va_info = read_in_dict(va_info_file)
        self.url = 'http://nlp.uniroma1.it/amuse-wsd/api/model'
        self.headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        self.language = language
        if filter_layer:
            layer_ckpt = plus+f"model/amuse/filter_layer_{language}"
            # this is a refinement layer computed on the notebook of this homework
            self.filter = torch.load(layer_ckpt)

    def use_AMuSE_WSD(self, sentence: Dict[str, List[str]]):
        # http://nlp.uniroma1.it/amuse-wsd/api-documentation
        stringa = " ".join(sentence["words"])
        dict =  [{"text": stringa, "lang" : self.language}]
        payload = json.dumps(dict)
        r = requests.post( self.url, data=payload, headers=self.headers)
        sol = r.json()[0]['tokens']
        senses = list()
        for i, n in enumerate(sentence["predicates"]):
            if n == 0:
                senses.append("_")
            else:
                bn_id = sol[i]["bnSynsetId"]
                if bn_id not in self.bn2va:
                    # if we have a "noun prediction" it's clearly not a verb
                    senses.append("_")
                    continue
                vf_from_bn = self.bn2va[bn_id]
                senses.append(self.use_filtered_predictions(vf_from_bn))
        return {"predicates" : senses}

    def use_filtered_predictions(self,vf_from_bn):
        if hasattr(self, 'filter') and self.va_info[vf_from_bn] in self.filter:
            return self.filter[self.va_info[vf_from_bn]]
        else:
            return self.va_info[vf_from_bn]

    def predict(self, sentences: Union[List[str], Dict[str, List[str]]], require_ids = False):
        """
        INPUT:
        {
            "words": [...], # SAME AS BEFORE
            "lemmas": [...], # SAME AS BEFORE
            "predicates":
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ]
        }
        OUTPUT:
        {
            "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
        }
        """
        if not require_ids:
            return self.use_AMuSE_WSD(sentences)
        # ELSE    
        predictions = dict()
        # need a copy to avoid side effects
        sentences2 = copy.deepcopy(sentences)
        for id in sentences2:
            sentences2[id]["predicates"] = [0 if i == '_' else 1 for i in sentences2[id]["predicates"]]
            # here as always "private" evaluation
            predictions[id] = self.use_AMuSE_WSD(sentences2[id])
        return predictions