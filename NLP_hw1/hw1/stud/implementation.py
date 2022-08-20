from typing import List
from model import Model
from .dataprocess import * #note relative import to work with docker
from .my_crf import CRF
from torch import nn
import json

class HParams():
    vocab_size = 0 #it will be modified by StudentModel with the imported vocabulary
    embedding_dim = 300
    hidden_dim = 270
    num_classes = 13 # number of different NER classes for this homework
    bidirectional = True 
    num_layers = 5
    dropout = 0.4
    trainable_embeddings = True 
    proj_size = 0
    scaling_classifier = 3
    model = None
    pretrained_path = None
    vocabulary_path = None
    embeddings_path = None

def build_model(device: str) -> Model:
    #I've used as my final model an ensemble of networks, some that use the CFR layer and some that don't. 
    hparams = HParams() #they are going to share most of the hparams
    hparams.vocabulary_path = 'model/glove.6B.300d_extended2.json' 
    hparams.model=NERModel
    hparams.pretrained_path = "model/NERModelv270e1_7454.pth"
    model1=StudentModel(device,hparams)
    hparams.pretrained_path = "model/NERModelv270e2_7531.pth"
    model2=StudentModel(device,hparams)
    hparams.pretrained_path = "model/NERModelv270e5_7431.pth"
    model5=StudentModel(device,hparams)
    
    hparams.model=NERModel_crf
    hparams.pretrained_path = "model/NERModelv270e3_crf7566.pth"
    model3=StudentModel(device,hparams)
    hparams.pretrained_path = "model/NERModelv270e4_crf7521.pth"
    model4=StudentModel(device,hparams)
    hparams.model=NERModel
    
    models=[model1,model2,model3,model4,model5]
    return Ensemble_model(models)

class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device:str,hparams:HParams):
        self.device=device
        assert(hparams.vocabulary_path!=None)
        with open(hparams.vocabulary_path, 'r') as file:
            self.vocab = json.load(file)
            print(len(self.vocab))
        hparams.vocab_size=len(self.vocab)
        self.model = hparams.model(hparams).to(device)
        if hparams.pretrained_path!=None:
            self.model.load_state_dict(torch.load(hparams.pretrained_path, map_location=device)) #weights of the best model so far

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.eval()
        window_size_ = 50 #!hyperparameter, use a large value to predict well
        nd=NERDataset(tokens,self.vocab,labels=None,window_size=window_size_)
        token_dataset=DataLoader(nd, batch_size=64, shuffle=False)
        res = list()
        for x in token_dataset:
            output=self.model.predict(x['inputs'].to(self.device))
            res+=sum(output,[]) #to flat the list
        res=nd.decode_output(res)
        predictions = list()
        #to keep the same order of tokens!
        for sentence in tokens:
            current_len=len(sentence)
            predict_sentence=list()
            for i in range(current_len):
                predict_sentence.append(res.pop(0))
            #we pop out the useless results computed associated to <pad> tokens
            if current_len%window_size_ != 0:
                for j in range(window_size_-(current_len%window_size_)):
                    res.pop(0)
            predictions.append(predict_sentence)
        assert(len(res) == 0) #res should be empty to perfectly match all the required tags
        return predictions

# this model is based on POSTaggerModel from notebook #6
class NERModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(NERModel, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings_path is not None:
            self.word_embedding.load_state_dict(torch.load(hparams.embeddings_path))
            if hparams.trainable_embeddings:
                for param in self.word_embedding.parameters(): #to freeze training parameters
                    param.requires_grad = False    

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first=True,
                            proj_size=hparams.proj_size)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier=nn.Linear(lstm_output_dim,hparams.num_classes)

    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        return self.classifier(o)
    def predict(self,x):
        out = self(x)
        return torch.argmax(out,-1).tolist()
#it's exactly like the above class, what changes is that we predict using Viterbi algorithm (implemented in CRF_decode).
#this was one of my last "extra" experiments so I did not changed all my code but I adapted the new one to my framework. 
class NERModel_crf(nn.Module):
    def __init__(self, hparams):
        super(NERModel_crf, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings_path is not None:
            self.word_embedding.load_state_dict(torch.load(hparams.embeddings_path))
            if hparams.trainable_embeddings:
                for param in self.word_embedding.parameters(): #to freeze training parameters
                    param.requires_grad = False    

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first=True,
                            proj_size=hparams.proj_size)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier=nn.Linear(lstm_output_dim,hparams.num_classes)
        self.crf=CRF(hparams.num_classes,batch_first=True)

    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        return self.classifier(o)
    
    def predict(self,x):
        out=self(x)
        return self.crf.decode(out)
    
class Ensemble_model(Model):
    """_summary_
        Generic class for "Model" models, it take as input a list of "Model" and predict using
        a majority voting with uniform weights for each of them.
    """
    def __init__(self,models:list):
        self.num_models=len(models)
        self.models=models
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        all_predictions=list() #list that will contains the predictions for each model
        for mod in self.models:
             all_predictions.append(mod.predict(tokens))
        ensemble_predict=list()
        for mixpred in zip(* all_predictions): #we zip the prediction of each sentence for each model
            tmp_pred=list()
            for i in range(len(mixpred[0])):
                #create the list with the  word predict for each model and predict with the most common result
                tmp=[mixpred[j][i] for j in range(self.num_models)]
                tmp_pred.append(max(tmp,key=tmp.count))
            ensemble_predict.append(tmp_pred)
        return ensemble_predict