import os
from tracemalloc import stop
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from src.utils import config
import torch.utils.data as data
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact", "oWant", "oEffect", "oReact"]
stop_words = stopwords.words("english")

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None

def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item):
    cs_list = {}
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list[rel] = cs_res
    return cs_list


def encode_ctx(vocab, items, data_dict, comet):
    for ctx in tqdm(items):
        ctx_list = []
        csk_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            csk_list.append(get_commonsense(comet, item))
        data_dict["context"].append(ctx_list)
        data_dict["utt_cs"].append(csk_list)

def encode(vocab, files):
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "utt_cs": [],
    }
    comet = Comet("./comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]
    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab


def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc_new.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
        
    for i in range(3):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    
    if config.csk_feature:
        csk_tra, csk_val, csk_tst = pickle.load(open('data/ED/csk_features.pkl', 'rb'), encoding='latin1')
        data_tra["csk_feature"] = csk_tra
        data_val["csk_feature"] = csk_val
        data_tst["csk_feature"] = csk_tst

    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"], item["x_cls_index"], item["user_cls_index"], item["user_mask"], item["agent_mask"], item["graph_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )

        if config.csk_feature:
            item["csk_feature"] = self.data["csk_feature"][index]
            item["x_intent"] = torch.FloatTensor(item["csk_feature"]["xIntent"])
            item["x_need"] = torch.FloatTensor(item["csk_feature"]["xNeed"])
            item["x_want"] = torch.FloatTensor(item["csk_feature"]["xWant"])
            item["x_effect"] = torch.FloatTensor(item["csk_feature"]["xEffect"])
            item["x_react"] = torch.FloatTensor(item["csk_feature"]["xReact"])
        else:
            item["cs_text"] = self.data["utt_cs"][index]
            item["x_intent_txt"] = [Know['xIntent'] for Know in item["cs_text"]]
            item["x_need_txt"] = [Know['xNeed'] for Know in item["cs_text"]]
            item["x_want_txt"] = [Know['xWant'] for Know in item["cs_text"]]
            item["x_effect_txt"] = [Know['xEffect'] for Know in item["cs_text"]]
            item["x_react_txt"] = [Know['xReact'] for Know in item["cs_text"]]

            item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
            item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
            item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
            item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
            item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        return item

    def construct_graph_adj(self, arr):
        pass

    def preprocess(self, arr, anw=False, cs=None, emo=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            sequence = []
            for sent in arr:
                csk = [config.CLS_idx] if cs != "react" else []
                for item in sent:
                    csk += [
                        self.vocab.word2index[word]
                        for word in item
                        if word in self.vocab.word2index and word not in ["to", "none"]
                    ]
                sequence.append(torch.tensor(csk, dtype=torch.long))
            sequence = pad_sequence(sequence, batch_first=True, padding_value=config.PAD_idx)
            return sequence
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

        else:
            x_dial = []
            x_mask = []
            x_cls_index = []
            user_cls_index = []
            if config.wo_dis_sel_oth:
                graph_mask = np.zeros((6*len(arr)+2, 6*len(arr)+2))
                agent_mask = np.zeros((6*len(arr)+4, 6*len(arr)+4)) # utt + csk + 4 states
                user_mask = np.zeros((6*len(arr)+4, 6*len(arr)+4))
            else:
                if config.dis_emo_cog:
                    if not config.wo_csk:
                        graph_mask = np.zeros((6*len(arr)+2, 6*len(arr)+2))
                        agent_mask = np.zeros((6*len(arr)+4, 6*len(arr)+4)) # utt + csk + 4 states
                        user_mask = np.zeros((6*len(arr)+4, 6*len(arr)+4))
                    else:
                        graph_mask = np.zeros((6*len(arr)+2, 6*len(arr)+2))
                        agent_mask = np.zeros((len(arr)+4, len(arr)+4)) # utt + csk + 4 states
                        user_mask = np.zeros((len(arr)+4, len(arr)+4))
                else:
                    agent_mask = np.zeros((6*len(arr)+2, 6*len(arr)+2)) # utt + csk + 2 states
                    user_mask = np.zeros((6*len(arr)+2, 6*len(arr)+2))

            csk_offset = len(arr)
            for i, sentence in enumerate(arr):
                # x_dial += [config.CLS_idx] + [
                #     self.vocab.word2index[word]
                #     if word in self.vocab.word2index
                #     else config.UNK_idx
                #     for word in sentence
                # ]
                if i % 2 == 0:
                    x_dial += [config.USR_idx] + [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                else:
                    x_dial += [config.SYS_idx] + [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                
                x_cls_index += [len(x_dial) - len(sentence) - 1]
                if i % 2 == 0:
                    user_cls_index += [x_cls_index[-1]]
                
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence) + 1)] # [config.CLS_idx] + [spk for _ in range(len(sentence))]

                if config.wo_dis_sel_oth:
                    graph_mask[-1][-1] = 1
                    graph_mask[-2][-2] = 1
                    if i % 2 == 0:
                        j = i
                        while j >= 0:
                            if j % 2 == 0:
                                graph_mask[i][j] = 1 # utt-connection
                                graph_mask[j][i] = 1
                                graph_mask[-2][j] = 1 # user state connection with utt
                                graph_mask[j][-2] = 1
                                graph_mask[-1][j] = 1 # user state connection with utt
                                graph_mask[j][-1] = 1
                            j -= 1
                        for j in range(5):
                            graph_mask[csk_offset+j][csk_offset+j] = 1 # csk self-loop
                            graph_mask[csk_offset+j][csk_offset+j] = 1
                            graph_mask[i][csk_offset+j] = 1 # csk-utt connection
                            graph_mask[csk_offset+j][i] = 1
                            if j == 4:
                                graph_mask[-1][csk_offset+j] = 1 # user emo state connection with csk
                                graph_mask[csk_offset+j][-1] = 1
                            else:
                                graph_mask[-2][csk_offset+j] = 1 # user cog state connection with csk
                                graph_mask[csk_offset+j][-2] = 1
                        csk_offset += 5
                    else:
                        j = i
                        while j >= 0:
                            if j % 2 != 0:
                                graph_mask[i][j] = 1 # utt-connection
                                graph_mask[j][i] = 1
                                graph_mask[-1][j] = 1 # agent emo state connection with utt
                                graph_mask[j][-1] = 1
                                graph_mask[-2][j] = 1 # agent cog state connection with utt
                                graph_mask[j][-2] = 1
                            j -= 1
                        for j in range(5):
                            graph_mask[csk_offset+j][csk_offset+j] = 1 # csk self-loop
                            graph_mask[csk_offset+j][csk_offset+j] = 1
                            graph_mask[i][csk_offset+j] = 1 # csk-utt connection
                            graph_mask[csk_offset+j][i] = 1
                            if j == 4:
                                graph_mask[-1][csk_offset+j] = 1 # agent emo state connection with csk
                                graph_mask[csk_offset+j][-1] = 1 
                            else:
                                graph_mask[-2][csk_offset+j] = 1 # agent cog state connection with csk
                                graph_mask[csk_offset+j][-2] = 1
                        csk_offset += 5
                else:
                    if config.dis_emo_cog:
                        agent_mask[-1][-1] = 1
                        agent_mask[-2][-2] = 1
                        user_mask[-3][-3] = 1
                        user_mask[-4][-4] = 1
                    else:
                        agent_mask[-1][-1] = 1
                        user_mask[-2][-2] = 1
                    if i % 2 == 0:
                        j = i
                        while j >= 0:
                            if j % 2 == 0:
                                user_mask[i][j] = 1 # utt-connection
                                user_mask[j][i] = 1
                                if config.dis_emo_cog:
                                    user_mask[-3][j] = 1 # user emo state connection with utt
                                    user_mask[j][-3] = 1
                                    user_mask[-4][j] = 1 # user cog state connection with utt
                                    user_mask[j][-4] = 1
                                else:
                                    user_mask[-2][j] = 1 # user state connection with utt
                                    user_mask[j][-2] = 1
                            j -= 1
                        if not config.wo_csk:
                            for j in range(5):
                                user_mask[csk_offset+j][csk_offset+j] = 1 # csk self-loop
                                user_mask[csk_offset+j][csk_offset+j] = 1
                                user_mask[i][csk_offset+j] = 1 # csk-utt connection
                                user_mask[csk_offset+j][i] = 1
                                if config.dis_emo_cog:
                                    if j == 4:
                                        user_mask[-3][csk_offset+j] = 1 # user emo state connection with csk
                                        user_mask[csk_offset+j][-3] = 1
                                    else:
                                        user_mask[-4][csk_offset+j] = 1 # user cog state connection with csk
                                        user_mask[csk_offset+j][-4] = 1
                                else:
                                    user_mask[-2][csk_offset+j] = 1 # user state connection with csk
                                    user_mask[csk_offset+j][-2] = 1
                            csk_offset += 5
                    else:
                        j = i
                        while j >= 0:
                            if j % 2 != 0:
                                agent_mask[i][j] = 1 # utt-connection
                                agent_mask[j][i] = 1
                                if config.dis_emo_cog:
                                    agent_mask[-1][j] = 1 # agent emo state connection with utt
                                    agent_mask[j][-1] = 1
                                    agent_mask[-2][j] = 1 # agent cog state connection with utt
                                    agent_mask[j][-2] = 1
                                else:
                                    agent_mask[-1][j] = 1 # agent state connection with utt
                                    agent_mask[j][-1] = 1
                            j -= 1
                        if not config.wo_csk:
                            for j in range(5):
                                agent_mask[csk_offset+j][csk_offset+j] = 1 # csk self-loop
                                agent_mask[csk_offset+j][csk_offset+j] = 1
                                agent_mask[i][csk_offset+j] = 1 # csk-utt connection
                                agent_mask[csk_offset+j][i] = 1
                                if config.dis_emo_cog:
                                    if j == 4:
                                        agent_mask[-1][csk_offset+j] = 1 # agent emo state connection with csk
                                        agent_mask[csk_offset+j][-1] = 1 
                                    else:
                                        agent_mask[-2][csk_offset+j] = 1 # agent cog state connection with csk
                                        agent_mask[csk_offset+j][-2] = 1
                                else:
                                    agent_mask[-1][csk_offset+j] = 1 # agent state connection with csk
                                    agent_mask[csk_offset+j][-1] = 1
                            csk_offset += 5
                
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask), torch.LongTensor(x_cls_index), torch.LongTensor(user_cls_index), torch.FloatTensor(user_mask), torch.FloatTensor(agent_mask), torch.FloatTensor(graph_mask)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    
    def pad_matrix(matrix, padding_index=0):
        max_len = max(i.size(0) for i in matrix)
        batch_matrix = []
        for item in matrix:
            item = item.numpy()
            batch_matrix.append(np.pad(item, ((0, max_len-len(item)), (0, max_len-len(item))), 'constant', constant_values=(padding_index, padding_index)))
        return torch.FloatTensor(batch_matrix)

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    x_cls_index = pad_sequence(item_info["x_cls_index"], True)
    user_cls_index = pad_sequence(item_info["user_cls_index"], True)
    user_mask = pad_matrix(item_info["user_mask"])
    agent_mask = pad_matrix(item_info["agent_mask"])
    graph_mask = pad_matrix(item_info["graph_mask"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    target_batch = target_batch.to(config.device)
    x_cls_index = x_cls_index.to(config.device)
    user_cls_index = user_cls_index.to(config.device)
    user_mask = user_mask.to(config.device)
    agent_mask = agent_mask.to(config.device)
    graph_mask = graph_mask.to(config.device)

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["x_cls_index"] = x_cls_index
    d["user_cls_index"] = user_cls_index
    d["user_mask"] = user_mask
    d["agent_mask"] = agent_mask
    d["graph_mask"] = graph_mask

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"] #
    if config.csk_feature:
        for r in relations:
            pad_batch = pad_sequence(item_info[r], batch_first=True, padding_value=config.PAD_idx).to(config.device)
            d[r] = pad_batch
    else:
        for r in relations:
            pad_batch = item_info[r]
            pad_batch = [item.to(config.device) for item in pad_batch]
            d[r] = pad_batch
            d[f"{r}_txt"] = item_info[f"{r}_txt"]

    return d


def prepare_data_seq(batch_size=32):

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )
