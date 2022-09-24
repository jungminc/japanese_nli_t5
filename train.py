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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.Normalizer import normalizer
from src.Antonym import antonym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = '/groups/gcb50246/jungminc/'
os.chdir("/home/ace14443ne/coliee4asqa")


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

    input_ids = tokenizer.batch_encode_plus([art + que for art, que in zip(articles, questions)], return_tensors='pt', padding=True).input_ids
    attention_mask = tokenizer.batch_encode_plus([art + que for art, que in zip(articles, questions)], return_tensors='pt', padding=True).attention_mask

    labels = torch.tensor([tokenizer('はい。').input_ids if l==1 else tokenizer('いいえ。').input_ids for l in labels])
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
                    shuffle=False
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
    Out = []
    Gold = []
    Logit = []
    model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            out = model.module.generate(input_ids) 
            logits = model(input_ids, labels=labels).logits
            logits = logits[:,1,[7,2090]] #7 corresponds with は, 2090 with いい
            logits = torch.nn.functional.log_softmax(logits, dim=2)
            for l in out:
                if l[2]!=7 and l[2]!=2090:
                    print("error!")
            out = [1 if l[2]==7 else 0 for l in out]
            pred = torch.argmax(logits, dim=1)
            labels = [1 if l[1]==7 else 0 for l in labels]
            Out.extend(out)
            Pred.extend(pred.tolist())
            Gold.extend(labels)
            Logit.extend(logits.tolist())
    acc = accuracy_score(Out, Gold)
    prec = precision_score(Out, Gold)
    recl = recall_score(Out, Gold)
    print(acc, prec, recl)
    return acc, Logit, Gold, Out




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

pretrained_name = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(pretrained_name)

indices = list(range(len(pseudo_labels)))
random.shuffle(indices)
pseudo_val_labels = [pseudo_labels[i] for i in indices[:int(len(indices)*0.1)]]
pseudo_val_questions = [pseudo_questions[i] for i in indices[:int(len(indices)*0.1)]]
pseudo_val_articles = [pseudo_articles[i] for i in indices[:int(len(indices)*0.1)]]

pseudo_train_labels = [pseudo_labels[i] for i in indices[int(len(indices)*0.1):]]
pseudo_train_questions = [pseudo_questions[i] for i in indices[int(len(indices)*0.1):]]
pseudo_train_articles = [pseudo_articles[i] for i in indices[int(len(indices)*0.1):]]

train_labels += pseudo_train_labels
train_articles += pseudo_train_articles
train_questions += pseudo_train_questions

val_labels += pseudo_val_labels
val_articles += pseudo_val_articles
val_questions += pseudo_val_questions

# train_labels += pseudo_labels
# train_articles += pseudo_articles
# train_questions += pseudo_questions

pretrained_name = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(pretrained_name)

softmax = nn.Softmax(dim=-1)
batch_size = 64
train_dataloader = make_dataloader(train_articles, train_questions, train_labels, batch_size, shuffle=True)
val_dataloader = make_dataloader(val_articles, val_questions, val_labels, batch_size, shuffle=False)
test_dataloader = make_dataloader(test_articles, test_questions, test_labels, batch_size, shuffle=False)
df = pd.read_csv('./data/val.tsv', sep='\t', names=column_names)
val_labels, val_articles, val_questions = list(map(int, df.label.to_list())), df.article.to_list(), df.question.to_list()
orig_val_dataloader = make_dataloader(val_articles, val_questions, val_labels, batch_size, shuffle=False)
acc, prec, recl = [], [], []
for i in range(10):
    model = T5ForConditionalGeneration.from_pretrained(pretrained_name, output_hidden_states=True)
    model = model.to(device)
    model = nn.DataParallel(model)
    early_stop = 5
    count = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 1e+6
    for epoch in range(30):
        train_loop(train_dataloader, model)
        val_loss = val_loop(val_dataloader, model)
        if val_loss < best_loss:
            torch.save(model.state_dict(), OUT_DIR + "t5base_{}".format(i+1))
            best_loss = val_loss
            count = 0
        else:
            count += 1
            if count == early_stop:
                break

    model.load_state_dict(torch.load(OUT_DIR + 't5base_{}'.format(i+1)))
    print("below test")
    _, Test_Logit, Test_Gold, Test_Out = test_loop(test_dataloader, model)
    _, Val_Logit, Val_Gold, Val_Out = test_loop(orig_val_dataloader, model)
    acc.append(accuracy_score(Test_Out, test_labels))
    prec.append(precision_score(Test_Out, test_labels))
    recl.append(recall_score(Test_Out, test_labels))
    torch.save((Val_Logit, Test_Logit), OUT_DIR + 'Logit_t5base_{}'.format(i+1))
print("acc")
print(torch.std_mean(torch.tensor(acc)))
print("prec")
print(torch.std_mean(torch.tensor(prec)))
print("recl")
print(torch.std_mean(torch.tensor(recl)))

    
