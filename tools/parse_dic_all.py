import os
base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "datas")
process_data_path = os.path.join(data_path,"NCLS-Processed")
which_lan = 'ZH2ENSUM'
process_data_zh2en_path = os.path.join(process_data_path,which_lan)
en_voc_dic_path = os.path.join(process_data_zh2en_path,"bert_en_vocab.txt")
zh_voc_dic_path = os.path.join(process_data_zh2en_path,"bert_zh_vocab.txt")
all_voc_dic_path = os.path.join(process_data_zh2en_path,"all_bert_voc_dic.txt")
with open(en_voc_dic_path,'r',encoding="utf-8") as f:
    en_dic = f.readlines()
    en_dic = [ item.strip() for item in en_dic]
with open(zh_voc_dic_path,'r',encoding="utf-8") as f:
    zh_dic = f.readlines()
    zh_dic = [item.strip() for item in zh_dic]
all_dic_origin = en_dic+zh_dic
all_dic = list(set(all_dic_origin))
all_dic.sort(key=all_dic_origin.index)
with open(all_voc_dic_path,'w',encoding="utf-8") as f:
    for dic in all_dic:
        f.write(f'{dic}\n')