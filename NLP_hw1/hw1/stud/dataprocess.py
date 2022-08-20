import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Any, Dict
import string
def filter_sentences(
        sentences: List[str],
        labels: List[str],
        vocab: dict,
        min_size: int,
        max_size: int,
        filter_dirty=False
    ):
    """
    This function filter out all the sentences that exceed the size bounds or that are usesless for the training, that means all "O" labels
    or the "non-O" labels are for OOV words (if filter_dirty actived).
    """
    new_sentences=list()
    new_labels=list()
    k=0
    for s,lab in zip(sentences,labels):
        if  len(s)>max_size or len(s)<min_size or (filter_dirty and \
            all((w not in vocab and w.lower() not in vocab) or l=='O' for w,l in zip(s,lab))):
            k+=1 #increase count of sentences removed
        else:
            new_sentences.append(s)
            new_labels.append(lab)
    print("filtered out {} sentences".format(k))
    return new_sentences,new_labels

#this function is based on POSTaggingDataset seen in notebook #6      
class NERDataset(Dataset):
    #static variable useful for decode and encode the output
    label_encoding={
        "B-PER": 0, "B-LOC": 1, "B-GRP": 2, "B-CORP": 3, "B-PROD": 4, "B-CW": 5,
        "I-PER": 6, "I-LOC": 7,"I-GRP": 8, "I-CORP": 9, "I-PROD": 10, "I-CW": 11, 
        "O": 12,"<PAD>":13,"<UNK>":14
        }   #note UNK should never be used for encodings in labels, just allows me to write only one time encode_text
    def __init__(self, 
                 sentences: List[List[str]],
                 vocab_sent: dict,
                 labels=None,
                 window_size:int=50, 
                 window_shift:int=-1, #if negative non-overlapping window
                 ):
        self.window_size = window_size
        self.window_shift = window_shift if window_shift > 0 else window_size
        sentences_ = self.create_windows(sentences,vocab_sent)
        labels_ = self.create_windows(labels,self.label_encoding) if labels is not None else None
        self.data = self.make_dataset_items(sentences_,labels_)

    def create_windows(self, sentences, embedding):
        data = []
        for sentence in sentences:
          for i in range(0, len(sentence), self.window_shift):
            window = sentence[i:i+self.window_size]
            if len(window) < self.window_size:
              window = window + [None]*(self.window_size - len(window)) #concatenate to a list, to avoid not windows_size window
            assert len(window) == self.window_size
            data.append(self.encode_text(window,embedding))
        return data
             
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def make_dataset_items(sentences: torch.tensor,labels: torch.tensor):
        data=list()
        if labels is not None:
            for s,l in zip(sentences,labels):
                data.append({'inputs':s,'outputs':l})
        else:
            for s in sentences:
                data.append({'inputs':s})
        return data
    @staticmethod
    def special_find(word:str, embedding:dict, punct=["'","!",":",",","+"]):
        """This function takes as input a word and embedding and a list of punctuation, if the word it's not in embedding
        it check if in this word there is a special character, if there is return the special word to try to embed"""
        to_ret="<UNK>"
        if word in embedding:
            to_ret=word
        elif word.lower() in embedding:
            to_ret=word.lower()
        else:
            for pun in punct:
                if word.find(pun)>0:
                    to_ret="<SPECIAL"+pun+">"
                    break
        return to_ret    
    @staticmethod
    def encode_text(sentence:list,embedding: dict):
        """
        returns the embedding of "sentence" according to "embedding" dict in a torch.tensor form
        """
        embed = list()
        for w in sentence:
            embed.append(embedding.get(NERDataset.special_find(w,embedding),embedding["<UNK>"]) if w is not None else embedding["<PAD>"])
        return torch.tensor(embed,dtype=torch.int64)
    
    @staticmethod
    def decode_output(outputs:List[int]):
        label_decoding=dict((NERDataset.label_encoding[k], k) for k in NERDataset.label_encoding)
        decode = list()
        for id in outputs:
            decode.append(label_decoding[id])
        return decode