import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Any, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import transformers_embedder as tre
import numpy as np
import spacy
from spacy.tokens import Doc
# all "package relative imports" here, to avoid repeat code in the notebook as I did for hw1
try:
    from .amuse import AMuSE_WSD_online
except:
    from amuse import AMuSE_WSD_online

class Dataset_Base(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Dataset_SRL_34(Dataset_Base):
    def __init__(self, sentences: Dict[str, List[str]], need_train: bool, language: str):
        # if the dataset is for a model that need_train we assume to have labels
        self.has_labels = need_train 
        self.language = language
        if self.has_labels:
            # with this function we create a dict to encode and decode easily labels of roles
            self.labels_to_id, self.id_to_labels = Dataset_SRL_34.create_labels_id_mapping_roles()
        self.data = self.make_data(sentences)

    @staticmethod
    def create_labels_id_mapping_roles():
        # these labels have been extracted studying the dataset from the notebook
        labels = ['agent', 'theme', 'beneficiary', 'patient', 'topic', 'goal', 'recipient', 
            'co-theme', 'result', 'stimulus', 'experiencer', 'destination', 'value', 'attribute', 
            'location', 'source', 'cause', 'co-agent', 'time', 'co-patient', 'product', 'purpose', 
            'instrument', 'extent', 'asset', 'material', '_']
        return {lab: i for i, lab in enumerate(labels)}, {i: lab for i, lab in enumerate(labels)}

    def make_data(self, sentences):
        data = list() 
        for ids in sentences:
            # we extract the position of the predicates
            predicate_positions = [i for i, p in enumerate(sentences[ids]["predicates"]) if p != '_' and p != 0]
            sentence_l = sentences[ids]["lemmas"] 
            for predicate_position in predicate_positions:
                item = dict()
                # here I build a tuple to attain the same input as proposed by "Shi - Lin 19" after the embedding
                # I have also added the predicate disambiguation of that value to check if it's possible to improve the result.
                item["input"] = (sentence_l, [sentence_l[predicate_position], sentences[ids]["predicates"][predicate_position]])
                if self.has_labels:
                    # the desired output are the labels already encoded for each role associated to the sentence[predicate_position]
                    # there is a problem in the french dataset since a label is attriute instead of attribute
                    item["role_labels"] = [self.labels_to_id[i] for i in sentences[ids]["roles"][predicate_position] if i != 'attriute']
                    # note that input and output have different sizes after the input embedding if a word_piece tokenizer is used
                    # (as in my case), but don't worry, the model will produce an average mean of piece_tokens so that the sizes
                    # will be compatible.
                data.append(item)               
        return data

class Dataset_SRL_234(Dataset_Base):
    def __init__(self, sentences: Dict[str, List[str]], need_train: bool, language: str, standard_dataset: bool = True):
        # if the dataset is for a model that need_train we assume to have labels
        self.has_labels = need_train 
        self.language = language
        self.standard_dataset = standard_dataset
        self.frame_to_id, self.id_to_frame= Dataset_SRL_234.create_frames_id_mapping()
        self.data = self.make_data(sentences)

    @staticmethod
    def create_frames_id_mapping():
        # these labels have been extracted studying the dataset from the notebook
        frames_in_dataset = ['_', 'ASK_REQUEST', 'BENEFIT_EXPLOIT', 'PLAN_SCHEDULE', 'CARRY-OUT-ACTION', 'ESTABLISH', 'SIMPLIFY', 'PROPOSE', 'TAKE-INTO-ACCOUNT_CONSIDER', 'BEGIN', 'CIRCULATE_SPREAD_DISTRIBUTE', 'REFER', 'SHOW', 'PRECLUDE_FORBID_EXPEL', 'VIOLATE', 'VERIFY', 'CAUSE-SMT', 'ABSTAIN_AVOID_REFRAIN', 'TRANSMIT', 'SEE', 'SUMMON', 'GUARANTEE_ENSURE_PROMISE', 'RECEIVE', 'INCREASE_ENLARGE_MULTIPLY', 'DECREE_DECLARE', 'PAY', 'CAUSE-MENTAL-STATE', 'CAGE_IMPRISON', 'HURT_HARM_ACHE', 'MOVE-BACK', 'EXIST_LIVE', 'CALCULATE_ESTIMATE', 'ATTRACT_SUCK', 'EXIST-WITH-FEATURE', 'INFORM', 'EXPLAIN', 'SPEAK', 'SEEM', 'MISS_OMIT_LACK', 'DECIDE_DETERMINE', 'ASSIGN-SMT-TO-SMN', 'FOLLOW_SUPPORT_SPONSOR_FUND', 'MOVE-ONESELF', 'WORSEN', 'AMELIORATE', 'AGREE_ACCEPT', 'MOVE-SOMETHING', 'PUT_APPLY_PLACE_PAVE', 'ADJUST_CORRECT', 'INCLUDE-AS', 'CONTINUE', 'SPEED-UP', 'LOAD_PROVIDE_CHARGE_FURNISH', 'REMEMBER', 'FINISH_CONCLUDE_END', 'REPEAT', 'HELP_HEAL_CARE_CURE', 'IMPLY', 'OPPOSE_REBEL_DISSENT', 'STRENGTHEN_MAKE-RESISTANT', 'AROUSE_WAKE_ENLIVEN', 'RECORD', 'INCITE_INDUCE', 'GIVE_GIFT', 'DESTROY', 'REQUIRE_NEED_WANT_HOPE', 'ANALYZE', 'COME-AFTER_FOLLOW-IN-TIME', 'BELIEVE', 'GO-FORWARD', 'CANCEL_ELIMINATE', 'RECOGNIZE_ADMIT_IDENTIFY', 'CHOOSE', 'REPRESENT', 'TREAT', 'OBLIGE_FORCE', 'STOP', 'REACT', 'HAPPEN_OCCUR', 'OVERCOME_SURPASS', 'AFFECT', 'CREATE_MATERIALIZE', 'ALLY_ASSOCIATE_MARRY', 'MANAGE', 'OPEN', 'ORIENT', 'ANSWER', 'INFLUENCE', 'COMBINE_MIX_UNITE', 'LEAD_GOVERN', 'STAY_DWELL', 'WELCOME', 'AMASS', 'PREPARE', 'ORGANIZE', 'HAVE-A-FUNCTION_SERVE', 'GIVE-UP_ABOLISH_ABANDON', 'SORT_CLASSIFY_ARRANGE', 'GIVE-BIRTH', 'PUBLISH', 'USE', 'POSSESS', 'BEHAVE', 'WORK', 'SUBJECTIVE-JUDGING', 'APPROVE_PRAISE', 'ATTEND', 'LEAVE_DEPART_RUN-AWAY', 'CATCH', 'OBEY', 'SATISFY_FULFILL', 'UNDERSTAND', 'ACHIEVE', 'TRY', 'ATTACH', 'INTERPRET', 'DELAY', 'REDUCE_DIMINISH', 'UNDERGO-EXPERIENCE', 'RETAIN_KEEP_SAVE-MONEY', 'ARRIVE', 'REFUSE', 'IMAGINE', 'HARMONIZE', 'PARTICIPATE', 'HIRE', 'RESULT_CONSEQUENCE', 'FOCUS', 'CONTAIN', 'MOUNT_ASSEMBLE_PRODUCE', 'PROVE', 'WRITE', 'RESTRAIN', 'TOLERATE', 'ACCOMPANY', 'DISCUSS', 'RESTORE-TO-PREVIOUS/INITIAL-STATE_UNDO_UNWIND', 'TEACH', 'CHANGE-APPEARANCE/STATE', 'INVERT_REVERSE', 'RELY', 'SIGNAL_INDICATE', 'LEARN', 'ACCUSE', 'PERFORM', 'AFFIRM', 'REMOVE_TAKE-AWAY_KIDNAP', 'WATCH_LOOK-OUT', 'GROUND_BASE_FOUND', 'LEAVE-BEHIND', 'FACE_CHALLENGE', 'CHANGE_SWITCH', 'SHARE', 'APPLY', 'ARGUE-IN-DEFENSE', 'DIRECT_AIM_MANEUVER', 'WAIT', 'HEAR_LISTEN', 'CONSIDER', 'LIKE', 'FIGHT', 'PROTECT', 'AUTHORIZE_ADMIT', 'DIVERSIFY', 'PRESERVE', 'LOCATE-IN-TIME_DATE', 'SEND', 'ORDER', 'SEARCH', 'REGRET_SORRY', 'EMPHASIZE', 'CELEBRATE_PARTY', 'TAKE-SHELTER', 'HOST_MEAL_INVITE', 'REPLACE', 'THINK', 'MEET', 'PERCEIVE', 'BREAK_DETERIORATE', 'JOIN_CONNECT', 'BORDER', 'FIND', 'KNOW', 'KILL', 'CHARGE', 'FAIL_LOSE', 'CRITICIZE', 'CITE', 'HIT', 'LIBERATE_ALLOW_AFFORD', 'BRING', 'DERIVE', 'JUSTIFY_EXCUSE', 'PERSUADE', 'REVEAL', 'DRIVE-BACK', 'TAKE', 'OBTAIN', 'LOSE', 'ADD', 'MATCH', 'CONSUME_SPEND', 'COMPARE', 'BEFRIEND', 'NAME', 'BE-LOCATED_BASE', 'OFFER', 'OVERLAP', 'CARRY_TRANSPORT', 'REACH', 'FILL', 'ENCLOSE_WRAP', 'DISBAND_BREAK-UP', 'COUNT', 'DEFEAT', 'CO-OPT', 'ENDANGER', 'PUNISH', 'TRANSLATE', 'SECURE_FASTEN_TIE', 'INSERT', 'REMAIN', 'BUY', 'STEAL_DEPRIVE', 'SETTLE_CONCILIATE', 'EXTEND', 'SUMMARIZE', 'PUBLICIZE', 'CORRELATE', 'SEPARATE_FILTER_DETACH', 'GROUP', 'COST', 'ATTACK_BOMB', 'WARN', 'NEGOTIATE', 'ENTER', 'LIE', 'SPEND-TIME_PASS-TIME', 'EMPTY_UNLOAD', 'INVERT_REVERSE-', 'EMIT', 'TURN_CHANGE-DIRECTION', 'SELL', 'GUESS', 'DISCARD', 'CONTRACT-AN-ILLNESS_INFECT', 'WASH_CLEAN', 'DROP', 'OPERATE', 'SHARPEN', 'REFLECT', 'COMPENSATE', 'ASCRIBE', 'LOWER', 'COPY', 'DEBASE_ADULTERATE', 'DISMISS_FIRE-SMN', 'COVER_SPREAD_SURMOUNT', 'MEASURE_EVALUATE', 'RESIGN_RETIRE', 'READ', 'DISTINGUISH_DIFFER', 'TRAVEL', 'RESIST', 'SHOOT_LAUNCH_PROPEL', 'BURDEN_BEAR', 'SOLVE', 'WIN', 'APPEAR', 'FOLLOW-IN-SPACE', 'PULL', 'PAINT', 'COME-FROM', 'VISIT', 'COOL', 'DOWNPLAY_HUMILIATE', 'CHASE', 'EMBELLISH', 'EARN', 'RAISE', 'PROMOTE', 'MEAN', 'EXHAUST', 'ABSORB', 'PRESS_PUSH_FOLD', 'LEND', 'SHAPE', 'PRINT', 'REPAIR_REMEDY', 'GROW_PLOW', 'QUARREL_POLEMICIZE', 'TAKE-A-SERVICE_RENT', 'COMPETE', 'DIVIDE', 'COMMUNICATE_CONTACT', 'FIT', 'EXEMPT', 'SLOW-DOWN', 'FLOW', 'RISK', 'METEOROLOGICAL', 'NOURISH_FEED', 'STABILIZE_SUPPORT-PHYSICALLY']
        return {lab: i for i, lab in enumerate(frames_in_dataset)}, {i: lab for i, lab in enumerate(frames_in_dataset)}

    def make_data(self, sentences):
        data = list() 
        if self.standard_dataset:
            # I load precomputed ones, my internet is slow and computing them takes to long
            if "1996/a/50/18_supp__323:5" in sentences or "1996/a/50/18_supp__323:5" in sentences:
                # I know that this idx is of an english training sentence
                frames = torch.load(f"../../model/amuse/prediction_words_new_{self.language}")
            else:
                frames = torch.load(f"../../model/amuse/prediction_words_dev_new_{self.language}")
        else:
            amuse = AMuSE_WSD_online(self.language)
            frames = amuse.predict(sentences, require_ids=True)
        #we will use this value as index for oov predicates
        for ids in sentences:
            # if ids ==  "2003/a/58/562_24:1":
            #     continue
            item = dict()
            item["input"] = sentences[ids]["words"]
            # we punt a random pred if OOV 
            item['frames'] = [self.frame_to_id.get(i, 1) for i in frames[ids]['predicates']]
            assert(len(item["input"]) == len(item["frames"]))
            if self.has_labels:
                item["labels"] = [self.frame_to_id.get(i, -100) for i in sentences[ids]['predicates']]       
            data.append(item)                      
        return data

class Dataset_SRL_1234(Dataset_Base):
    def __init__(self, sentences: Dict[str, List[str]], need_train: bool, language: str):
        # if the dataset is for a model that need_train we assume to have labels
        self.has_labels = need_train 
        self.language = language
        self.tags_to_id, self.id_to_tags= Dataset_SRL_1234.create_ptag_id_mapping()
        self.data = self.make_data(sentences)

    @staticmethod
    def create_ptag_id_mapping():
        # these labels have been extracted studying the dataset from the notebook
        ptags = ['NOUN', 'ADV', 'VERB', 'SCONJ', 'DET', 'ADJ', 'ADP', 'PUNCT', 'PROPN', 'PART', 
                'NUM', 'CCONJ', 'AUX', 'PRON', 'SYM', 'X', 'INTJ']
        return {lab: i for i, lab in enumerate(ptags)}, {i: lab for i, lab in enumerate(ptags)}

    def make_data(self, sentences):
        data = list() 
        taggers = {"EN":"en_core_web_sm", "ES":"es_core_news_sm", "FR":"fr_core_news_sm"}
        nlp = spacy.load(taggers[self.language])
        for ids in sentences:
            item = dict()
            ###
            # NOTE: there is at least a sentence 2003/a/58/562_24:1 with a '' token, I think this is a bug but since
            # I'm not allowed to change the dataset I have to manually replace it with a " "
            sentence_w = sentences[ids]["words"]
            try:
                sentence = Doc(nlp.vocab, sentence_w)
            except:
                sentence_w = [w if w != '' else " " for w in sentence_w]
                sentence = Doc(nlp.vocab, sentence_w)
            doc = nlp(sentence)
            pos_tag = list()
            for token in doc:
                # we do not have spaces in our tags in normal cases, it could be a missclassification
                pos_tag.append(self.tags_to_id[token.pos_] if token.pos_ != "SPACE" else self.tags_to_id["PUNCT"])
            assert(len(sentence_w) == len(pos_tag))
            item["input"] = sentence_w
            item['pos_tags'] = pos_tag
            if self.has_labels:
                item["labels"] = [0 if i == '_' else 1 for i in sentences[ids]["predicates"]]       
            data.append(item)                      
        return data

class SRL_DataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict, task: str, language: str, sentences: Dict[str, List[str]], sentences_test: Dict[str, List[str]] = None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.sentences = sentences
        self.sentences_test = sentences_test
        assert(task in ["1234", "234", "34"])
        self.task = task
        self.language = language
        self.collates = {"34": self.collate_fn_34, "234": self.collate_fn_234, "1234": self.collate_fn_1234}

    def setup(self, stage: Optional[str] = None) -> None:
        
        if self.task == "34":
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.language_model_name)
            DATASET = Dataset_SRL_34
        elif self.task == "234":
            DATASET = Dataset_SRL_234
            self.tokenizer = tre.Tokenizer(self.hparams.language_model_name)
        else:
            self.tokenizer = tre.Tokenizer(self.hparams.language_model_name)
            DATASET = Dataset_SRL_1234
        self.data_train = DATASET(self.sentences, self.hparams.need_train, self.language)

        if self.sentences_test: 
            self.data_test = DATASET(self.sentences_test, self.hparams.need_train, self.language)

    def train_dataloader(self):
        # change collate based on the task
        return DataLoader(
                self.data_train, 
                batch_size = self.hparams.batch_size, 
                shuffle = True,
                num_workers = self.hparams.n_cpu,
                collate_fn = self.collates[self.task],
                pin_memory = True,
                persistent_workers = True
            )

    def val_dataloader(self):
        # change collate based on the task
        return DataLoader(
                    self.data_test, 
                    batch_size = self.hparams.batch_size, 
                    shuffle = False,
                    num_workers = self.hparams.n_cpu,
                    collate_fn = self.collates[self.task],
                    pin_memory = True,
                    persistent_workers = True
                )
    # here we define our collate function to apply the padding
    def collate_fn_34(self, batch) -> Dict[str, torch.Tensor]:
        batch_out = self.tokenizer.batch_encode_plus(
            [sentence["input"] for sentence in batch],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
        )
        word_ids = list()
        labels = list()
        for i, sentence in enumerate(batch):
            w_id = np.array(batch_out.word_ids(batch_index=i), dtype=np.float64)
            # since w_id contains None values we want to remove them to have the conversion in tensor
            # we'll do so by creating a "special_token index" that is +1 higher than the last word token
            special_idx = np.nanmax(w_id) + 1
            w_id[np.isnan(w_id)] = special_idx
            # we need to mask vectors belonging to the "second" sentence (the one made to produce the embedding) 
            # they all will get a value different from all the other tokens so it will be easy to remove
            w_id[batch_out["token_type_ids"][i]] = special_idx
            word_ids.append(w_id)
            if self.hparams.need_train:
                labels.append(sentence["role_labels"])
        if self.hparams.need_train:
            labels = pad_sequence(
                    [torch.as_tensor(label) for label in labels],
                    batch_first=True,
                    padding_value=-100
                )
            batch_out["labels"] = torch.as_tensor(labels)
        # np conversion of the list to speedup the tensor creation
        batch_out["word_id"] = torch.as_tensor(np.array(word_ids), dtype=torch.long) 
        return batch_out

    def collate_fn_234(self, batch) -> Dict[str, torch.Tensor]:
        return SRL_DataModule.common_collate(self, batch, "frames", self.hparams.n_frames)

    def collate_fn_1234(self, batch) -> Dict[str, torch.Tensor]:
        return SRL_DataModule.common_collate(self, batch, "pos_tags", self.hparams.pos_tag_tokens)

    @staticmethod
    def common_collate(self, batch, padkey1: str, padding_value1: int):
        # the last 2 collate_fn shares almost everything, so I collect the differences in this function
        batch_out = self.tokenizer(
            [sentence["input"] for sentence in batch],
            return_tensors = True,
            padding = True,
            is_split_into_words = True
        )
        batch_out[padkey1] = pad_sequence(
                                    [torch.as_tensor(item[padkey1]) for item in batch],
                                    batch_first=True,
                                    padding_value=padding_value1
                                )
        if self.hparams.need_train:
            batch_out["labels"] = pad_sequence(
                                    [torch.as_tensor(item["labels"]) for item in batch],
                                    batch_first=True,
                                    padding_value=-100
                                )
        return batch_out