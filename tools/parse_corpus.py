from lxml import etree
import os
import spacy
from transformers import BertTokenizer
import collections
which_lan = 'ZH2ENSUM'
# en_nlp = spacy.load('en_core_web_sm')
# zh_nlp = spacy.load('zh_core_web_sm')
en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
zh_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "datas")
process_data_path = os.path.join(data_path,"NCLS-Processed")
if not os.path.exists(process_data_path):
    os.mkdir(process_data_path)
process_data_zh2en_path = os.path.join(process_data_path,which_lan)
if not os.path.exists(process_data_zh2en_path):
    os.mkdir(process_data_zh2en_path)

part_name = "train"
file_path = os.path.join(data_path,f"NCLS-Data/{which_lan}/{which_lan}_{part_name}.txt")

processed_en_path = os.path.join(process_data_zh2en_path,f"{which_lan}_{part_name}_en.txt")
processed_zh_path = os.path.join(process_data_zh2en_path,f"{which_lan}_{part_name}_zh.txt")
en_voc_dic_path = os.path.join(process_data_zh2en_path,"en_voc_dic.txt")
zh_voc_dic_path = os.path.join(process_data_zh2en_path,"zh_voc_dic.txt")
all_voc_dic_path = os.path.join(process_data_zh2en_path,"all_voc_dic.txt")
en_lines = []
zh_lines = []
with open(file_path,'r',encoding="utf-8") as f:
    xml_tree = etree.HTML(f.read())
docs = xml_tree[0]
# if not os.path.exists(en_voc_dic_path) or not  os.path.exists(zh_voc_dic_path):
#     en_dic = set()
#     zh_dic = set()
# else:
#     with open(en_voc_dic_path,'r',encoding="utf-8") as f:
#         en_dic = f.readlines()
#         en_dic = set([ item.strip() for item in en_dic])
#     with open(zh_voc_dic_path,'r',encoding="utf-8") as f:
#         zh_dic = f.readlines()
#         zh_dic = set([item.strip() for item in zh_dic])
en_token_list = []
zh_token_list = []
for doc in docs:
    en_line = None
    zh_line = None
    for element in doc.iter("en-summary","back-translated-zh-summary"):
        if element.tag == "en-summary":
            en_line = element.text.strip()
        elif element.tag == "back-translated-zh-summary":
            zh_line = element.text.strip()
    if en_line and zh_line:
        # en_doc = en_nlp(en_line)
        # zh_doc = zh_nlp(zh_line)
        # en_tokens = [ tok.text for tok in en_doc]
        # zh_tokens = [ tok.text for tok in zh_doc]
        en_tokens = en_tokenizer.tokenize(en_line)
        zh_tokens = zh_tokenizer.tokenize(zh_line)
        # en_token_list.extend(en_tokens)
        # zh_token_list.extend(zh_tokens)

        en_line = ' '.join(en_tokens)
        zh_line = ' '.join(zh_tokens)
        en_lines.append(en_line)
        zh_lines.append(zh_line)

        print(en_tokens,zh_tokens)
        print(en_line,zh_line)

# en_token_counter = collections.Counter(en_token_list)
# zh_token_counter = collections.Counter(zh_token_list)
# print(en_token_counter.most_common())
# en_tokens = set([ key for key,item in en_token_counter.most_common() if item > 1])
# zh_tokens = set([ key for key,item in zh_token_counter.most_common() if item > 1])
# print(en_tokens)
# en_dic.update(en_tokens)
# zh_dic.update(zh_tokens)

# with open(en_voc_dic_path,'w',encoding="utf-8") as f:
#     for dic in en_dic:
#         f.write(f'{dic}\n')
# with open(zh_voc_dic_path,'w',encoding="utf-8") as f:
#     for dic in zh_dic:
#         f.write(f'{dic}\n')
#
# with open(all_voc_dic_path,'w',encoding="utf-8") as f:
#     for dic in list(en_dic)+list(zh_dic):
#         f.write(f'{dic}\n')

with open(processed_en_path,'w',encoding="utf-8") as f:
    for line in en_lines:
        f.write(line+'\n')
with open(processed_zh_path,'w',encoding="utf-8") as f:
    for line in zh_lines:
        f.write(line+'\n')