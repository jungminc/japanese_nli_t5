from transformers import BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, T5Model, BertModel, BertJapaneseTokenizer
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.chdir("/cl/work/jungmin-c/COLIEE2021statute_data-Japanese/")



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



def make_dataloader(articles, questions, labels, batch_size, shuffle):
    questions = [q.replace('、', '，') for q in questions]

    input_ids = tokenizer.batch_encode_plus([art+tokenizer.eos_token+que for art, que in zip(articles, questions)], return_tensors='pt', padding=True).input_ids
    attention_mask = tokenizer.batch_encode_plus([art+tokenizer.eos_token+que for art, que in zip(articles, questions)], return_tensors='pt', padding=True).attention_mask

    labels = torch.tensor([tokenizer('はい', add_special_tokens=False).input_ids if l==1 else tokenizer('いいえ', add_special_tokens=False).input_ids for l in labels])
    dataset = TensorDataset(input_ids, attention_mask, labels)


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
                    batch_size = batch_size,
                )

    return dataloader


def train_loop(dataloader, model):

    Loss = list()
    model.train() # 訓練モードで実行
    
    for i, batch in enumerate(dataloader):
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss = torch.mean(loss)
        Loss.append(loss.item())
        # loss = loss / args.accumulation_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if (i + 1) % args.accumulation_size == 0:
        #     if args.grad_clip > 0.:
        #         torch.nn.utils.clip_grad_norm_(
        #             model.parameters(), max_norm=args.grad_clip, norm_type=2
        #         )
        #     optimizer.step()
        #     optimizer.zero_grad()

    Loss = sum(Loss) / len(Loss)
    print(Loss)
    return Loss

def val_loop(dataloader, model):
    Loss = list()
    model.eval() 
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss = torch.mean(loss)
            Loss.append(loss.item())
    Loss = sum(Loss) / len(Loss)
    print("val loss is {}".format(Loss))
    return Loss

def test_loop(dataloader, model):
    Pred = []
    Gold = []
    model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            out = model.module.generate(input_ids, max_length=5, num_beams=5)
            for l in out:
                if l[2]!=7 and l[2]!=2090:
                    print("error!")
            out = [1 if l[2]==7 else 0 for l in out]
            labels = [1 if l[1]==7 else 0 for l in labels]
            Pred.extend(out)
            Gold.extend(labels)
    acc = accuracy_score(Pred, Gold)
    print(acc)
    return acc





questions, articles, labels = get_questions_with_articles_jp("./train")
test_questions, test_articles, test_labels = get_questions_with_articles_jp("./test")

questions_indices = list(range(len(questions)))
t_size = int(len(questions)*0.9)
random.shuffle(questions_indices)
t_indices, v_indices = questions_indices[:t_size], questions_indices[t_size:]
train_questions, train_articles, train_labels = [questions[i] for i in t_indices], [articles[i] for i in t_indices], [labels[i] for i in t_indices]
val_questions, val_articles, val_labels = [questions[i] for i in v_indices], [articles[i] for i in v_indices], [labels[i] for i in v_indices]



train_questions_pos, train_articles_pos, train_labels_pos = [train_questions[i] for i in range(len(train_questions)) if train_labels[i]==1], [train_articles[i] for i in range(len(train_questions)) if train_labels[i]==1], [train_labels[i] for i in range(len(train_questions)) if train_labels[i]==1]
val_questions_pos, val_articles_pos, val_labels_pos = [val_questions[i] for i in range(len(val_questions)) if val_labels[i]==1], [val_articles[i] for i in range(len(val_questions)) if val_labels[i]==1], [val_labels[i] for i in range(len(val_questions)) if val_labels[i]==1]

pretrained_name = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(pretrained_name)
model = T5ForConditionalGeneration.from_pretrained(pretrained_name, output_hidden_states=True)
model = model.to(device)
model = nn.DataParallel(model)

len_truncate = 8
batch_size = 32
train_dataloader = make_dataloader(train_articles, train_questions, train_labels, batch_size, shuffle=True)
val_dataloader = make_dataloader(val_articles, val_questions, val_labels, batch_size, shuffle=False)
test_dataloader = make_dataloader(test_articles, test_questions, test_labels, batch_size, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_loss = 1e+6
for epoch in range(20):
    train_loop(train_dataloader, model)
    val_loss = val_loop(val_dataloader, model)
    if val_loss < best_loss:
        torch.save(model.state_dict(), "coliee4asqa_wo_tsugihagi")
        best_loss = val_loss

model.load_state_dict(torch.load('coliee4asqa_wo_tsugihagi'))
acc = test_loop(test_dataloader, model)

