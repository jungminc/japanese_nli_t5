from transformers import BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, T5Model, BertModel, BertJapaneseTokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
import glob, os, re, sys, csv
import xml.etree.ElementTree as ET
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import CosineSimilarity
import MeCab
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.Normalizer import normalizer
from src.Antonym import antonym

from argparse import ArgumentParser
from logging import getLogger, INFO, DEBUG, Formatter, StreamHandler, FileHandler
import copy
import itertools

from bert_score import score
# Difference from inui5 is that this one tries to incorporate omitted characters like A B, ...

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.chdir("/cl/work/jungmin-c/COLIEE2021statute_data-Japanese/")

OUT_DIR = '/groups/gcb50246/jungminc/'
os.chdir("/home/ace14443ne/coliee4asqa/")
def decorate_logger(args, logger):
    """Decorate logger. 
       Stream for debug and File for experimental logs.
    """
    logger.setLevel(INFO)
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.log_path != "":
        f_handler = FileHandler(filename=args.log_path, mode="w", encoding="utf-8")
        f_handler.setLevel(INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger

def _mean_pooling(model_output_last_hidden_state, attention_mask):
    token_embeddings = model_output_last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_questions_with_articles_jp(directory):
    questions = []
    articles =[]
    labels = []
    q_id = -1
    for file in glob.glob(os.path.join(directory, '*.xml')):
        with open(file, encoding="utf-8") as f:
            xml = f.read()
            root = ET.fromstring(xml)
            for child in root:
                #q = tagger.parse(child[1].text.strip())
                q_id += 1
                aux = "[CLS]"
                q = child[1].text.strip().replace('\n','')
                questions.append(q)
                art_text = re.split('\n|\u3000| ',child[0].text)
                cleaned_art_text = ""
                l = 1 if child.attrib['label']=='Y' else 0
                labels.append(l)
                for i in range(len(art_text)):
                    if art_text[i] in ["", " ", "　"] or re.match(r"（.*）|^[１２３４５６７８９]|^第[一二三四五六七八九十百]+.{0,5}$", art_text[i]):
                        continue
                    normalized = re.sub(r"（.*）|^[１２３４５６７８９]|^第[一二三四五六七八九十百]+.{0,5}$", '', art_text[i])
                    cleaned_art_text += normalized + " "
                articles.append(cleaned_art_text[:-1])
                if cleaned_art_text=='':
                    print('here')

    return questions, articles, labels

def clean_questions(questions):
    new_questions = []
    for q in questions:
        if "各記述" not in q:
            new_questions.append(q)
            continue
        sentences = [s+"。" for s in q.split("。") if s]
        new_q = ""
        for s in sentences:
            if "各記述" in s:
                pass
            else:
                s = re.sub(r"（解答欄は.{0,5}）", "", s)
                s = re.sub(r"^[A-Z]", "", s)
                s = re.sub(r"^[１２３４５６７８９1-9]", "", s)
                s = re.sub(r"^(\.|\. |．|． )", "", s)
                new_q += s
        new_questions.append(new_q)
    return new_questions


def get_art_hier(path_to_articles):
    art_to_hier = {}
    with open(path_to_articles, encoding="utf-8") as f:
        lines = f.readlines()
    articles = {}
    art_to_id = {}
    hen_id=-1
    shou_id=-1
    setsu_id=-1
    meta = ""
    for i, line in enumerate(lines):
        hen = re.match(r'^第.{1,6}編', lines[i])
        shou = re.match(r'^第.{1,6}章', lines[i])
        setsu = re.match(r'^第.{1,6}節', lines[i])
        m = re.match(r'^第.{1,6}(条|条の.{1,4})( |　)', lines[i])
        if hen:
            hen_str = hen.group()
            hen_id += 1
            shou_id = -1
            setsu_id = -1
        if shou:
            shou_str = shou.group()
            shou_id += 1
            setsu_id = -1
        if setsu:
            setsu_str = setsu.group()
            setsu_id += 1
        if m:
            j = i + 1
            # while j < len(lines)-2 and not re.match(r'(^（.*）$)|(^第(?!.*条).*$)',lines[j]):
            while j < len(lines)-2 and not re.match(r'^（.*）$|^第.{1,6}(編|章|節|条|条の.{1,4}( |　|から|及び|))',lines[j]):
            # while j < len(lines)-2 and not re.match(r'^\(.*\)$|^Part.*$|^Chapter.*$|^Section.*$|^Article.*$',lines[j]):
                j += 1
            meta = re.match(r'^\(.*\)$', lines[i-1]).group() if re.match(r'^\(.*\)$', lines[i-1]) else meta
            lines_to_include = [meta + ' '] + lines[i:j] if meta else lines[i:j]
            art = ''.join([item for li in lines_to_include for item in li])
            if '削除' in art:
                continue
            id = len(art_to_hier)
            art_to_id[m.group()] = id
            articles[id] = hen_str + ' ' + shou_str + ' ' + setsu_str + ' ' + art if 'setsu_str' in locals() else hen_str + shou_str + art
            art_to_hier[id] = (hen_id, shou_id, setsu_id)
    hen_id, shou_id, setsu_id = 0, 0, 0
    new_art_to_hier = {}
    new_art_to_hier[0] = (0, 0, 0)
    for id in range(1,len(art_to_hier)):
        if art_to_hier[id][2]!= art_to_hier[id-1][2]:
            setsu_id+=1
            new_art_to_hier[id] = (art_to_hier[id][0], art_to_hier[id][1], setsu_id)
        else:
            new_art_to_hier[id] = (art_to_hier[id][0], art_to_hier[id][1], setsu_id)
    return articles, art_to_id, new_art_to_hier



def get_articles(path):
    articles = []

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        prev = i
        if re.match(r'^民法', lines[i]):
            i += 1
            continue
        if '削除' in lines[i]:
            i += 1
            continue
        if re.match(r'^（.*）$', lines[i]):
            i += 1
            continue
        if re.match(r'^第.{1,6}[編章節款目]', lines[i]):
            i += 1
            continue

        if re.match(r'第.{1,6}(条|条の.{1,4})', lines[i]):
            lines[i] = re.sub(r'第.{1,6}(条|条の.{1,4})( |　)', "", lines[i])
            j = i + 1
            while j < len(lines) and re.match(r'^[一二三四五六七八九十イロハニホヘト]', lines[j]):
                j += 1
            articles.append(" ".join(lines[i:j]))
            i = j
            continue
        if re.match(r'^[１２３４５６７８９]{1,2}　', lines[i]):
            lines[i] = re.sub(r'[１２３４５６７８９]{1,2}　', "", lines[i])
            j = i + 1
            while j < len(lines) and re.match(r'^[一二三四五六七八九十イロハニホヘト]', lines[j]):
                j += 1
            articles.append(" ".join(lines[i:j]))
            i = j
            continue
        if i==prev:
            print('something')

    return articles



def make_article_dataloader(articles):
    truncated, last = [a[:a.rfind('、')+1] for a in articles] , [a[a.rfind('、')+1:] for a in articles]
    prompt_input_ids = []
    prompt_attention_mask = []
    answer_input_ids = []
    
    for trun, las in zip(truncated, last):
        if len(las)<3:
            trun, las = trun[:-len_truncate], trun[-len_truncate:] + las
        # if 2 in art_input_ids:
        #     print('detected unk in art')
        #     print(art_input_ids)
        #     print(art)
        whole_input_ids = tokenizer.encode_plus(trun+las+tokenizer.eos_token, return_tensors='pt', add_special_tokens=False).input_ids[0]
        trun_input_ids = tokenizer.encode_plus(trun, return_tensors='pt', add_special_tokens=False).input_ids[0]
        las_input_ids = whole_input_ids[trun_input_ids.size(0):torch.count_nonzero(whole_input_ids)]
        # if 2 in que_input_ids:
        #     print('detected unk in que')
        #     print(que_input_ids)
        #     print(que)

        # if 2 in last_input_ids:
        #     print('detected unk in last')
        #     print(last_input_ids)
        #     print(que)

        prompt_input_ids.append(trun_input_ids[-512:])
        # prompt_attention_mask.append(torch.cat([art_attention_mask[:512-que_attention_mask.size(0)-last_input_ids.size(0)], que_attention_mask]))
        answer_input_ids.append(las_input_ids)
        

    prompt_input_ids = pad_sequence(prompt_input_ids, batch_first=True)
    # prompt_attention_mask = torch.where(prompt_input_ids==0, 0, prompt_input_ids)
    answer_input_ids = pad_sequence(answer_input_ids, batch_first=True)

    hinting_input_ids = []
    hinting_answer_input_ids = []
    for i, q in enumerate(prompt_input_ids):
        for j in range(torch.count_nonzero(answer_input_ids[i])):
            # hinting_input_ids.append(torch.cat([prompt_input_ids[i][:prompt_attention_mask[i].sum()-1], answer_input_ids[i][:j]]))
            hinting_input_ids.append(torch.cat([prompt_input_ids[i][:torch.count_nonzero(prompt_input_ids[i])], answer_input_ids[i][:j]]))
            hinting_answer_input_ids.append(answer_input_ids[i][j:])
    #一番最初のhinting_answer_input_idsの長さ - 1 個，次のexampleをとる．
    # When doing inference, see how different amount of hinting affects the final prediction of the 
    hinting_input_ids = pad_sequence(hinting_input_ids, batch_first=True)
    hinting_attention_mask = torch.where(hinting_input_ids!=0, 1, 0)
    hinting_answer_input_ids = pad_sequence(hinting_answer_input_ids, batch_first=True)

    dataset = TensorDataset(hinting_input_ids, hinting_attention_mask, hinting_answer_input_ids)
    dataloader = DataLoader(
                dataset,  
                sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                batch_size = batch_size,
                num_workers = 4
            )
    return dataloader

def replace_names(questions):
    japanese_names = ['佐藤', '鈴木', '高橋', '田中', '渡辺', '伊藤', '山本', '中村', '小林', '加藤'] 
    pattern = re.compile(r'[ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]')
    new_questions = []
    for q in questions:
        match = pattern.findall(q)
        match = set(match)
        count = 0
        name_map = {}
        for k in match:
            name_map[k] = japanese_names[count]
            count += 1
        for k in match:
            q = q.replace(k, name_map[k])
        new_questions.append(q)
    return new_questions

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.classifier = nn.Linear(3 * 768, 2)
        # self.classifier2 = nn.Linear(768, 768)
        # self.classifier3 = nn.Linear(768, 2)
        # self.dropout = nn.Dropout()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, emb_gol, emb_gen, labels):
        features = torch.cat([emb_gol, emb_gen, torch.abs(emb_gol - emb_gen)], dim=1)
        output = self.classifier(features)
        # output = self.dropout(output)
        # output = self.classifier2(output)
        # output = self.dropout(output)
        # output = self.classifier3(output)
        loss = self.loss_fn(output, labels)
        return loss, output

def make_hinting_dataloader(articles, questions, labels, batch_size, shuffle):
    questions = [q.replace('、', '，') for q in questions]

    questions_truncated, questions_last = [q[:q.rfind('，')+1] for q in questions] , [q[q.rfind('，')+1:] for q in questions]

    prompt_input_ids = []
    prompt_attention_mask = []
    answer_input_ids = []

    for art, que, last in zip(articles, questions_truncated, questions_last):
        if len(last)<3:
            que, last = que[:-len_truncate], que[-len_truncate:] + last
        art_input_ids = tokenizer.encode_plus(art+tokenizer.eos_token, return_tensors='pt', add_special_tokens=False).input_ids[0]
        # if 2 in art_input_ids:
        #     print('detected unk in art')
        #     print(art_input_ids)
        #     print(art)
        que_whole_input_ids = tokenizer.encode_plus(que+last+tokenizer.eos_token, return_tensors='pt', add_special_tokens=False).input_ids[0]
        que_input_ids = tokenizer.encode_plus(que, return_tensors='pt', add_special_tokens=False).input_ids[0]
        last_input_ids = que_whole_input_ids[que_input_ids.size(0):]
        # if 2 in que_input_ids:
        #     print('detected unk in que')
        #     print(que_input_ids)
        #     print(que)

        # if 2 in last_input_ids:
        #     print('detected unk in last')
        #     print(last_input_ids)
        #     print(que)

        prompt_input_ids.append(torch.cat([art_input_ids[:512-que_input_ids.size(0)-last_input_ids.size(0)], que_input_ids]))
        # prompt_attention_mask.append(torch.cat([art_attention_mask[:512-que_attention_mask.size(0)-last_input_ids.size(0)], que_attention_mask]))
        answer_input_ids.append(last_input_ids)
        

    prompt_input_ids = pad_sequence(prompt_input_ids, batch_first=True)
    # prompt_attention_mask = torch.where(prompt_input_ids==0, 0, prompt_input_ids)
    answer_input_ids = pad_sequence(answer_input_ids, batch_first=True)

    hinting_input_ids = []
    hinting_answer_input_ids = []
    new_labels = []
    for i, q in enumerate(prompt_input_ids):
        for j in range(torch.count_nonzero(answer_input_ids[i])):
            # hinting_input_ids.append(torch.cat([prompt_input_ids[i][:prompt_attention_mask[i].sum()-1], answer_input_ids[i][:j]]))
            hinting_input_ids.append(torch.cat([prompt_input_ids[i][:torch.count_nonzero(prompt_input_ids[i])], answer_input_ids[i][:j]]))
            hinting_answer_input_ids.append(answer_input_ids[i][j:])
        new_labels.extend([labels[i]]*(torch.count_nonzero(answer_input_ids[i])))
    #一番最初のhinting_answer_input_idsの長さ - 1 個，次のexampleをとる．
    # When doing inference, see how different amount of hinting affects the final prediction of the 
    hinting_input_ids = pad_sequence(hinting_input_ids, batch_first=True)
    hinting_attention_mask = torch.where(hinting_input_ids!=0, 1, 0)
    hinting_answer_input_ids = pad_sequence(hinting_answer_input_ids, batch_first=True)
    labels = torch.tensor(new_labels)

    dataset = TensorDataset(hinting_input_ids, hinting_attention_mask, \
                            hinting_answer_input_ids, \
                            labels)
    if shuffle:
        dataloader = DataLoader(
                    dataset,  
                    sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                    batch_size = batch_size,
                    num_workers = 4
                )
    else:
        dataloader = DataLoader(
                    dataset,  
                    shuffle=False,
                    batch_size = batch_size,
                )


    return dataloader




# def make_perp_dataloader(questions, generated, batch_size, shuffle):

#     questions = [q.replace('、', '，') for q in questions]

#     questions_truncated, questions_last = [q[:q.rfind('，')+1] for q in questions] , [q[q.rfind('，')+1:] for q in questions]

#     generated_input_ids = []
#     for que, gen, last in zip(questions_truncated, generated, questions_last):
#         if len(last)<3:
#             que, last = que[:-len_truncate], que[-len_truncate:] + last

#         que_input_ids = tokenizer.encode_plus(que, return_tensors='pt', add_special_tokens=False).input_ids[0]
#         gen_input_ids = tokenizer.encode_plus(gen, return_tensors='pt', add_special_tokens=False).input_ids[0]

#         generated_input_ids.append(torch.cat([que_input_ids, gen_input_ids]))

#     generated_input_ids = pad_sequence(generated_input_ids, batch_first=True)
#     generated_attention_mask = torch.where(generated_input_ids!=0, 1, 0)

#     original_input_ids = tokenizer.batch_encode_plus(questions, return_tensors='pt', add_special_tokens=False, padding=True).input_ids
#     original_attention_mask = torch.where(original_input_ids!=0, 1, 0)

#     dataset = TensorDataset(generated_input_ids, generated_attention_mask, original_input_ids, original_attention_mask)

#     if shuffle:
#         dataloader = DataLoader(
#                     dataset,  
#                     sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
#                     batch_size = batch_size,
#                     num_workers = 4
#                 )
#     else:
#         dataloader = DataLoader(
#                     dataset,  
#                     shuffle=False,
#                     batch_size = batch_size,
#                 )

#     return dataloader

def make_perp_dataloader(generated, gold, batch_size, shuffle):

    generated_input_ids = tokenizer.batch_encode_plus(generated, return_tensors='pt', add_special_tokens=True, padding=True).input_ids
    generated_attention_mask = torch.where(generated_input_ids!=0, 1, 0)

    original_input_ids = tokenizer.batch_encode_plus(gold, return_tensors='pt', add_special_tokens=True, padding=True).input_ids
    original_attention_mask = torch.where(original_input_ids!=0, 1, 0)

    dataset = TensorDataset(generated_input_ids, generated_attention_mask, original_input_ids, original_attention_mask)

    if shuffle:
        dataloader = DataLoader(
                    dataset,  
                    sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                    batch_size = batch_size,
                    num_workers = 4
                )
    else:
        dataloader = DataLoader(
                    dataset,  
                    shuffle=False,
                    batch_size = batch_size,
                )

    return dataloader

def make_hinting_test_dataloader(articles, questions, labels, batch_size, shuffle, len_hint=0):
    questions = [q.replace('、', '，') for q in questions]

    questions_truncated, questions_last = [q[:q.rfind('，')+1] for q in questions] , [q[q.rfind('，')+1:] for q in questions]

    prompt_input_ids = []
    answer_input_ids = []

    for art, que, last in zip(articles, questions_truncated, questions_last):
        if len(last)<3:
            que, last = que[:-len_truncate], que[-len_truncate:] + last
        art_input_ids = tokenizer.encode_plus(art+tokenizer.eos_token, return_tensors='pt', add_special_tokens=False).input_ids[0]
        # if 2 in art_input_ids:
        #     print('detected unk in art')
        #     print(art_input_ids)
        #     print(art)
        que_whole_input_ids = tokenizer.encode_plus(que+last+tokenizer.eos_token, return_tensors='pt', add_special_tokens=False).input_ids[0]
        que_input_ids = tokenizer.encode_plus(que, return_tensors='pt', add_special_tokens=False).input_ids[0]
        last_input_ids = que_whole_input_ids[que_input_ids.size(0):]
        # if 2 in que_input_ids:
        #     print('detected unk in que')
        #     print(que_input_ids)
        #     print(que)

        # if 2 in last_input_ids:
        #     print('detected unk in last')
        #     print(last_input_ids)
        #     print(que)

        prompt_input_ids.append(torch.cat([art_input_ids[:512-que_input_ids.size(0)-last_input_ids.size(0)], que_input_ids]))
        # prompt_attention_mask.append(torch.cat([art_attention_mask[:512-que_attention_mask.size(0)-last_input_ids.size(0)], que_attention_mask]))
        answer_input_ids.append(last_input_ids)
        

    prompt_input_ids = pad_sequence(prompt_input_ids, batch_first=True)
    # prompt_attention_mask = torch.where(prompt_input_ids==0, 0, prompt_input_ids)
    answer_input_ids = pad_sequence(answer_input_ids, batch_first=True)

    hinting_input_ids = []
    hinting_answer_input_ids = []
    new_labels = []
    for i, q in enumerate(prompt_input_ids):
        hinting_input_ids.append(torch.cat([prompt_input_ids[i][:torch.count_nonzero(prompt_input_ids[i])], answer_input_ids[i][:len_hint]]))
        hinting_answer_input_ids.append(answer_input_ids[i][len_hint:])
    #一番最初のhinting_answer_input_idsの長さ - 1 個，次のexampleをとる．
    # When doing inference, see how different amount of hinting affects the final prediction of the 
    hinting_input_ids = pad_sequence(hinting_input_ids, batch_first=True)
    hinting_attention_mask = torch.where(hinting_input_ids!=0, 1, 0)
    hinting_answer_input_ids = pad_sequence(hinting_answer_input_ids, batch_first=True)
    labels = torch.tensor(labels)
    dataset = TensorDataset(hinting_input_ids, hinting_attention_mask, \
                            hinting_answer_input_ids, \
                            labels)
    if shuffle:
        dataloader = DataLoader(
                    dataset,  
                    sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                    batch_size = batch_size,
                    num_workers = 4
                )
    else:
        dataloader = DataLoader(
                    dataset,  
                    shuffle=False,
                    batch_size = batch_size,
                )


    return dataloader

def make_dataloader(articles, questions, labels, batch_size, shuffle):
    questions = [q.replace('、', '，') for q in questions]

    questions_truncated, questions_last = [q[:q.rfind('，')+1] for q in questions] , [q[q.rfind('，')+1:] for q in questions]

    prompt_input_ids = []
    prompt_attention_mask = []
    answer_input_ids = []

    for art, que, last in zip(articles, questions_truncated, questions_last):
        if len(last)<3:
            que, last = que[:-len_truncate], que[-len_truncate:] + last
        art_input_ids = tokenizer.encode_plus(art+tokenizer.eos_token, return_tensors='pt').input_ids[0]
        art_attention_mask = tokenizer.encode_plus(art, return_tensors='pt').attention_mask[0]
        que_input_ids = tokenizer.encode_plus(que, return_tensors='pt').input_ids[0]
        que_attention_mask = tokenizer.encode_plus(que, return_tensors='pt').attention_mask[0]
        last_input_ids = tokenizer.encode_plus(last, return_tensors='pt').input_ids[0]

        prompt_input_ids.append(torch.cat([art_input_ids[:512-que_input_ids.size(0)-last_input_ids.size(0)], que_input_ids]))
        prompt_attention_mask.append(torch.cat([art_attention_mask[:512-que_attention_mask.size(0)-last_input_ids.size(0)], que_attention_mask]))
        answer_input_ids.append(last_input_ids)

    prompt_input_ids = pad_sequence(prompt_input_ids, batch_first=True)
    prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True)
    answer_input_ids = pad_sequence(answer_input_ids, batch_first=True)

    labels = torch.tensor(labels)

    dataset = TensorDataset(prompt_input_ids, prompt_attention_mask, \
                            answer_input_ids, labels)
    if shuffle:
        dataloader = DataLoader(
                    dataset,  
                    sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                    batch_size = batch_size,
                    num_workers = 4
                )
    else:
        dataloader = DataLoader(
                    dataset,  
                    shuffle=False,
                    batch_size = batch_size,
                )

    return dataloader





def train_loop(dataloader, model):

    Loss = list()
    model.train() # 訓練モードで実行
    
    for i, batch in enumerate(dataloader):
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        last_tokens = batch[2].to(device) 
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=last_tokens)
        loss = out.loss
        loss = torch.mean(loss)
        Loss.append(loss.item())
        loss = loss / args.accumulation_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % args.accumulation_size == 0:
            if args.grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip, norm_type=2
                )
            optimizer.step()
            optimizer.zero_grad()

    Loss = sum(Loss) / len(Loss)
    print(Loss)
    return Loss

def val_loop(dataloader, model):
    Loss = list()
    model.eval() 
    for i, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        last_tokens = batch[2].to(device) 
        labels = batch[3].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=last_tokens)
        loss = out.loss
        loss = torch.mean(loss)
        Loss.append(loss.item())
    Loss = sum(Loss) / len(Loss)
    print("val loss is {}".format(Loss))
    return Loss

def generate_2st_data(dataloader, model):
    Gen = []
    Gold = []
    gold_labels = []
    model.eval() # 訓練モードで実行

    for i, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        last_tokens = batch[2].to(device) 
        labels = batch[3]

        out = model.module.generate(input_ids, max_length=last_tokens.size(1))

        if out[0].size(0) != 0:
            for i in range(out.size(0)):
                generated = tokenizer.decode(out[i], skip_special_tokens=True)
                Gen.append(generated)
                gold_tokens = tokenizer.decode(last_tokens[i], skip_special_tokens=True)

                Gold.append(gold_tokens)
                gold_labels.append(labels[i].item())

    return Gen, Gold, gold_labels


def make_2st_dataloder(Gen, Gold, labels, batch_size, shuffle):
    iis = tokenizer.batch_encode_plus([gen + tokenizer.eos_token + gol for gen, gol in zip(Gen, Gold)], padding=True, return_tensors="pt").input_ids
    ams = tokenizer.batch_encode_plus([gen + tokenizer.eos_token + gol for gen, gol in zip(Gen, Gold)], padding=True, return_tensors="pt").attention_mask

    labels = torch.tensor(labels)

    dataset = TensorDataset(iis, ams, labels)
    if shuffle:
        dataloader = DataLoader(dataset,
                            sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                            batch_size = batch_size,
                            num_workers = 4
                            )
    else:
        dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size = batch_size,
                            )
    return dataloader
    

def train_2st_loop(dataloader, model, classifier, optimizer):
    Loss = []
    model.train()
    classifier.train()

    for batch in dataloader:
        iis = batch[0].to(device)
        ams = batch[1].to(device)
        labels = batch[2]

        out = model.module.encoder(input_ids=iis, attention_mask=ams).last_hidden_state
        emb = _mean_pooling(out, ams)

        labels = torch.tensor([[1.,0.] if l==0 else [0., 1.] for l in labels]).to(device)
        # loss = criterion(emb_gen, emb_gol, labels)
        logits = classifier(emb)
        loss = criterion(logits, labels)
        loss.backward()
        Loss.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    Loss = sum(Loss) / len(Loss)
    print("train loss is {}".format(Loss))
    return Loss


def val_2st_loop(dataloader, model, classifier):
    Loss = []
    model.eval()
    classifier.eval()
    for batch in dataloader:
        with torch.no_grad():

            iis = batch[0].to(device)
            ams = batch[1].to(device)
            labels = batch[2]

            out = model.module.encoder(input_ids=iis, attention_mask=ams).last_hidden_state
            emb = _mean_pooling(out, ams)

            labels = torch.tensor([[1.,0.] if l==0 else [0., 1.] for l in labels]).to(device)
            logits = classifier(emb)

            loss = criterion(logits, labels)
            # loss = torch.mean(loss)

            Loss.append(loss.item())

    Loss = sum(Loss) / len(Loss)
    print("val_loss is {}".format(Loss))
    return Loss

def test_2st_loop(dataloader, model, th=None):
    model.eval()

    gold_labels = []
    Pred = []
    Logits = []
    for batch in dataloader:
        with torch.no_grad():

            iis = batch[0].to(device)
            ams = batch[1].to(device)
            labels = batch[2]
            gold_labels.extend(labels.tolist())
            out = model.module.encoder(input_ids=iis, attention_mask=ams).last_hidden_state
            emb = _mean_pooling(out, ams)

            logits = classifier(emb)
            preds = torch.argmax(logits, dim=1).tolist()
            Logits.extend(logits.tolist())
            
            Pred.extend(preds)

    print(accuracy_score(Pred, gold_labels))
    return Logits
    
def compute_similarity_within_section(articles_dict, art_to_hier, model):
    groups = []
    cur_id = 0
    tmp = []
    for a in art_to_hier:
        if art_to_hier[a][-1] != cur_id:
            cur_id += 1
            groups.append(tmp)
            tmp = []
        else:
            tmp.append(a)
    pairs = []
    for group in groups:
        combs = [[articles_dict[a], articles_dict[b]] for a,b in itertools.combinations(group, 2)]
        pairs.extend(combs)
    sent_a, sent_b = zip(*pairs)
    iis_a = tokenizer.batch_encode_plus(sent_a, padding=True, return_tensors="pt").input_ids
    ams_a = tokenizer.batch_encode_plus(sent_a, padding=True, return_tensors="pt").attention_mask
    iis_b = tokenizer.batch_encode_plus(sent_b, padding=True, return_tensors="pt").input_ids
    ams_b = tokenizer.batch_encode_plus(sent_b, padding=True, return_tensors="pt").attention_mask
    dataset = TensorDataset(iis_a, ams_a, iis_b, ams_b)
    dataloader = DataLoader(
                dataset,  
                shuffle=False,
                batch_size = batch_size,
                num_workers = 4
            )
    model.eval()
    sim = []
    for batch in dataloader:
        with torch.no_grad():
            iis_a, ams_a, iis_b, ams_b = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            out_a = model.module.encoder(input_ids=iis_a, attention_mask=ams_a).last_hidden_state
            emb_a = _mean_pooling(out_a, ams_a)
            out_b = model.module.encoder(input_ids=iis_b, attention_mask=ams_b).last_hidden_state
            emb_b = _mean_pooling(out_b, ams_b)
            sim.extend(cos_sim(emb_a, emb_b).tolist())
    print(sum(sim)/len(sim))

    return

def compute_similarity_between_question_article(pos_questions, pos_articles, model):
    iis_q = tokenizer.batch_encode_plus(pos_questions, padding=True, return_tensors="pt").input_ids
    ams_q = tokenizer.batch_encode_plus(pos_questions, padding=True, return_tensors="pt").attention_mask
    iis_a = tokenizer.batch_encode_plus(pos_articles, padding=True, return_tensors="pt").input_ids
    ams_a = tokenizer.batch_encode_plus(pos_articles, padding=True, return_tensors="pt").attention_mask
    dataset = TensorDataset(iis_q, ams_q, iis_a, ams_a)
    dataloader = DataLoader(
                dataset,  
                shuffle=False,
                batch_size = batch_size,
                num_workers = 4
            )
    model.eval()
    sim = []
    for batch in dataloader:
        with torch.no_grad():
            iis_q, ams_q, iis_a, ams_a = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            out_q = model.module.encoder(input_ids=iis_q, attention_mask=ams_q).last_hidden_state
            emb_q = _mean_pooling(out_q, ams_q)
            out_a = model.module.encoder(input_ids=iis_a, attention_mask=ams_a).last_hidden_state
            emb_a = _mean_pooling(out_a, ams_a)
            sim.extend(cos_sim(emb_q, emb_a).tolist())
    print(sum(sim)/len(sim))        
    return

def compute_perplexity(dataloader, model):
    model.eval()
    perp_scores_gen = []
    perp_scores_ori = [] 
    for batch in dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        input_ids_ori = batch[2].to(device)
        attention_mask_ori = batch[3].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            outputs_ori = model(input_ids=input_ids_ori, attention_mask=attention_mask_ori, labels=input_ids_ori)
        shift_logits = outputs.logits[:, :-1, :].contiguous() 
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        batch_size, seq_len = shift_labels.shape
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(batch_size, seq_len)
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)

        perp_scores_gen.extend(torch.exp(loss))

        shift_logits_ori = outputs_ori.logits[:, :-1, :].contiguous() 
        shift_labels_ori = input_ids_ori[:, 1:].contiguous()
        shift_mask_ori = attention_mask_ori[:, 1:].contiguous()
        batch_size_ori, seq_len_ori = shift_labels_ori.shape
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ori = loss_fn(shift_logits_ori.view(-1, shift_logits_ori.size(-1)), shift_labels_ori.view(-1)).view(batch_size_ori, seq_len_ori)
        loss_ori = (loss_ori * shift_mask_ori).sum(dim=1) / shift_mask_ori.sum(dim=1)

        perp_scores_gen.extend(torch.exp(loss))
        perp_scores_ori.extend(torch.exp(loss_ori))

    return sum(perp_scores_gen) / len(perp_scores_gen), sum(perp_scores_ori) / len(perp_scores_ori)



def compute_bert_score(gold, gen):
    Precision, Recall, F1 = score(gen, gold, lang="ja", verbose=True)
    return Precision.numpy().tolist(), Recall.numpy().tolist(), F1.numpy().tolist()


def compute_bleu(Gold, Generated):
    from torchtext.data.metrics import bleu_score
    wakati = MeCab.Tagger('-Owakati')
    Gold_w, Generated_w = [], []
    for gol, gen in zip(Gold, Generated):
        gol_w = wakati.parse(gol).strip().split()
        gen_w = wakati.parse(gen).strip().split()
        Gold_w.append([gol_w])
        Generated_w.append(gen_w)
    score = bleu_score(Generated_w, Gold_w)
    return score


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_in', default=None, type=str)
    parser.add_argument('--model_out', default="ModelOut", type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--pretrained1', default = "sonoisa/t5-base-japanese", type=str)
    parser.add_argument('--pretrained2', default = "cl-tohoku/bert-base-japanese", type=str)

    parser.add_argument('--max_epoch1', default=10, type=int)
    parser.add_argument('--max_epoch2', default=3, type=int)

    parser.add_argument('--lr0', default=1e-4, type=float)
    parser.add_argument('--lr1', default=1e-4, type=float)
    parser.add_argument('--lr2', default=1e-5, type=float)
    parser.add_argument('--accumulation_size', default=1, type=int)
    parser.add_argument('--grad_clip', default=0., type=float)

    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--train1', default=False, action='store_true')
    parser.add_argument('--train2', default=False, action='store_true')
    parser.add_argument('--test', default=False,  action='store_true')
    parser.add_argument('--version', default='ver1', type=str)

    parser.add_argument('--log_path', default="", type=str)

    args = parser.parse_args()

    logger = getLogger()
    logger = decorate_logger(args, logger)
    logger.info(args)
    logger.info(" ".join(sys.argv))


    seed = args.seed if args.seed != 0 else random.randint(0,2**16)
    torch.manual_seed(seed)
    random.seed(seed)
    print("seed is {}".format(seed))
    
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained1)

    column_names = ["label", "article", "question"]
    df = pd.read_csv('./data/train.tsv', sep='\t', names=column_names)
    train_labels, train_articles, train_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()
    df = pd.read_csv('./data/val.tsv', sep='\t', names=column_names)
    val_labels, val_articles, val_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()
    df = pd.read_csv('./data/test.tsv', sep='\t', names=column_names)
    test_labels, test_articles, test_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()
    df = pd.read_csv('./data/augmentation.tsv', sep='\t', names=column_names)
    pseudo_labels, pseudo_articles, pseudo_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()
    df = pd.read_csv('./data/tsugihagi_augmentation.tsv', sep='\t', names=column_names)
    tsugihagi_labels, tsugihagi_articles, tsugihagi_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()
    df = pd.read_csv('./data/augmentation_prev.tsv', sep='\t', names=column_names)
    pseudo_prev_labels, pseudo_prev_articles, pseudo_prev_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()

    train_questions = clean_questions(train_questions)
    train_questions = replace_names(train_questions)
    val_questions = clean_questions(val_questions)
    val_questions = replace_names(val_questions)
    test_questions = clean_questions(test_questions)
    test_questions = replace_names(test_questions)

    len_truncate = 8
    batch_size = args.batch_size

    Train_Gen, Train_Gold, train_gold_labels, Val_Gen, Val_Gold, val_gold_labels, Test_Gen, Test_Gold, test_gold_labels = torch.load(OUT_DIR + args.model_out + '_' +args.version)

    test_pos_indices = [i for i in range(len(test_labels)) if test_labels[i]==1]
    test_questions_pos = [test_questions[i] for i in test_pos_indices]
    Test_Gen_pos = [Test_Gen[i] for i in test_pos_indices]
    Test_Gold_pos = [Test_Gold[i] for i in test_pos_indices]

    test_neg_indices = [i for i in range(len(test_labels)) if test_labels[i]==0]
    test_questions_neg = [test_questions[i] for i in test_neg_indices]
    Test_Gen_neg = [Test_Gen[i] for i in test_neg_indices]
    Test_Gold_neg = [Test_Gold[i] for i in test_neg_indices]



    perp_dataloader = make_perp_dataloader(test_questions_pos, Test_Gen_pos, batch_size, shuffle=False)
    model_id = "rinna/japanese-gpt2-medium"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id)
    perp = compute_perplexity(perp_dataloader, model)

    print(perp)
    #(tensor(116418.2266), tensor(3566499.2500))
    bert_score = compute_bert_score(Test_Gold_pos, Test_Gen_pos)[0]
    print(sum(bert_score)/len(bert_score))
    # 0.8861354542405981
    bleu = compute_bleu(Test_Gold_pos, Test_Gen_pos)
    print(bleu)
    # 0.5157862549223442
