import json
import os
import numpy as np
import pandas as pd 
from datasets import load_dataset

class CulturePark:
    def __init__(self, path, language, text_key, label_key, label_true, label_false):
        self.__path = path
        self.__language = language
        self.__text_key = text_key
        self.__label_key = label_key
        self.__label_true = label_true
        self.__label_false = label_false

    def read_data(self):
        with open(self.__path, "rt") as f:
            data = [json.loads(line) for line in f]

            if isinstance(self.__label_true, set) and isinstance(self.__label_false, set):
                data = [(
                        x[self.__text_key],
                        1 if x[self.__label_key] in self.__label_true else (
                                0 if x[self.__label_key] in self.__label_false else -1
                        )
                    ) for x in data]
                
            elif isinstance(self.__label_true, set):
                data = [(
                        x[self.__text_key],
                        1 if x[self.__label_key] in self.__label_true else (
                                0 if x[self.__label_key] == self.__label_false else -1
                        )
                    ) for x in data]
                
            elif isinstance(self.__label_false, set):
                data = [(
                        x[self.__text_key],
                        1 if x[self.__label_key] == self.__label_true else (
                                0 if x[self.__label_key] in self.__label_false else -1
                        )
                    ) for x in data]
                
            else:
                data = [(
                        x[self.__text_key],
                        1 if x[self.__label_key] == self.__label_true else (
                                0 if x[self.__label_key] == self.__label_false else -1
                        )
                    ) for x in data]

        for datum in data:
            if datum[1] == -1:
                print(f"Unknown label detected: {self.get_path()}")
                break

        return data
    
    def get_language(self):
        return self.__language
    
    def get_path(self):
        return self.__path

               
class MultiHate:
    def __init__(self, path, language, anno_path, id):
        
        self.path = path 
        self.language = language
        self.anno_path  = anno_path
        self.annotations = pd.read_csv(self.anno_path).values
        self.id = id

    def read_data(self):
        texts = pd.read_csv(self.path)

        data = []
        for meme_id, text in zip(texts['Meme ID'], texts['Translation']):
            data.append((text, int(self.annotations[meme_id, self.id])))

        for datum in data:
            if datum[1] == -1:
                print(f"Unknown label detected: {self.get_path()}")
                break

        return data

    def get_language(self):
        return self.language
    
    def get_path(self):
        return self.path


class DETOX:
    def __init__(self, path, language, data, text_key, label_key, label_true, label_false):
        
        self.path = path 
        self.language = language

        self.data  = data

        self.text_key    = text_key
        self.label_key   = label_key
        self.label_true  = label_true
        self.label_false = label_false


    def read_data(self):

        data = []
        for item in self.data:
            data.append((item[self.text_key], int(item[self.label_key])))

        for datum in data:
            if datum[1] == -1:
                print(f"Unknown label detected: {self.get_path()}")
                break

        return data

    def get_language(self):
        return self.language
    
    def get_path(self):
        return self.path

class DETOX_EXPLAIN:
    def __init__(self, path, language, data, text_key, label_key, label_true, label_false):
        
        self.path = path 
        self.language = language

        self.data  = data

        self.text_key    = text_key
        self.label_key   = label_key
        self.label_true  = label_true
        self.label_false = label_false


    def read_data(self):

        data = []
        for item in self.data:
            label = 1
            if item[self.label_key] == 'Low':
                label = 0
            data.append((item[self.text_key], label))

        for datum in data:
            if datum[1] == -1:
                print(f"Unknown label detected: {self.get_path()}")
                break

        return data

    def get_language(self):
        return self.language
    
    def get_path(self):
        return self.path



def statistics(data):
    hash_lan = {}
    hash_pos = {}
    hash_neg = {}

    for entry in data:
        lang  = entry['language']
        label = entry['label']

        if lang not in hash_lan:
            hash_lan[lang] = 1
        else:
            hash_lan[lang] += 1

        if label:
            if lang not in hash_pos:
                hash_pos[lang] = 1
            else:
                hash_pos[lang] += 1
        else:
            if lang not in hash_neg:
                hash_neg[lang] = 1
            else:
                hash_neg[lang] += 1
    


    print('all  : ',hash_lan)
    print('off  : ',hash_pos)
    print('noff : ',hash_neg)

    return 

if __name__ == "__main__":
    
    textdetox_multilingual = load_dataset("textdetox/multilingual_toxicity_dataset")
    textdetox_explain = load_dataset("textdetox/multilingual_toxicity_explained")
    print(textdetox_multilingual['es'])
    print(textdetox_explain['es'])

    SOURCES = [
        # SPANISH DATA: ONLY PICH AGGRESSIVENESS FOR DETOXIS 2021.
        CulturePark("culture-data/Spanish/AMI IberEval 2018_offens/data-2.jsonl", "spanish", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Spanish/DETOXIS 2021/aggressiveness.jsonl", "spanish", "data", "label", "1", "0"),
        CulturePark("culture-data/Spanish/HateEval 2019_HS/data-2.jsonl", "spanish", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Spanish/HaterNet_HS/data-2.jsonl", "spanish", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Spanish/MEX-A3T_offens/data-2.jsonl", "spanish", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Spanish/OffendES_offens/data-2.jsonl", "spanish", "data", "label", "OFF", "NOT"),
        MultiHate("multihate-data/captions/es.csv","spanish", "multihate-data/final_annotations.csv", 3),
        DETOX("detox-es.csv","spanish", textdetox_multilingual['es'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-es.csv","spanish", textdetox_explain['es'], 'Sentence', 'Toxicity Level', 1, 0),

        # PORTUGUESE
        CulturePark("culture-data/Portuguese/HateBR/data-2.jsonl", "portuguese", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Portuguese/OffComBR/data.jsonl", "portuguese", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Portuguese/ToLD-Br/homophobia.jsonl", "portuguese", "data", "label", "1", "0"),
        CulturePark("culture-data/Portuguese/ToLD-Br/insult.jsonl", "portuguese", "data", "label", "1", "0"),
        CulturePark("culture-data/Portuguese/ToLD-Br/misogyny.jsonl", "portuguese", "data", "label", "1", "0"),

        # TURKISH: DISCARD FINE_GRAINED INFO
        CulturePark("culture-data/Turkey/ATC/fold_0_test.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/offensDetect-kaggle2/test.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        # CulturePark("data/Turkey/offenseCorpus/offens_fine-graind.jsonl", "turkey", "tweet", "label"),
        CulturePark("culture-data/Turkey/offenseCorpus/offens.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/OffensEval2020/OffensEval.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/offenssDetect-kaggle/turkish_tweets_2020.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        #CulturePark("culture-data/Turkey/TurkishSpam/trspam.jsonl", "turkey", "tweet", "label", "Spam", "Ham"),

        # KOREAN
        CulturePark("culture-data/Korean/AbuseEval/data-2.jsonl", "korean", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Korean/CADD/data-2.jsonl", "korean", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Korean/K-MHaS/data-2.jsonl", "korean", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Korean/Korean-Hate-Speech-Detection/data-2.jsonl", "korean", "data", "label", "HS", {"NOT_HS", ""}),
        CulturePark("culture-data/Korean/KoreanHateSpeechdataset/data-2.jsonl", "korean", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Korean/Waseem/data-2.jsonl", "korean", "data", "label", "OFF", "NOT"),

        # GREECE
        #CulturePark("culture-data/Greece/gazzetta/G-TEST-S-preprocessed.jsonl", "greece", "tweet", "label", "OFF", "NOT"),
        #CulturePark("culture-data/Greece/OffensEval2020/OffensEval.jsonl", "greece", "tweet", "label", "OFF", "NOT"),

        # GERMANY: ONLY PICK ONE OF THE hatespeech refugees
        CulturePark("culture-data/Germany/GermEval/germeval2018.jsonl", "germany", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Germany/HASOC/hate_off_detect.jsonl", "germany", "tweet", "label", "HOF", "NOT"),
        #CulturePark("data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_1.jsonl", "germany", "tweet", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_2.jsonl", "germany", "tweet", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Germany/MHC/hatecheck_cases_final_german.jsonl", "germany", "tweet", "label", "HS", "NOT_HS"),
        MultiHate("multihate-data/captions/de.csv","germany", "multihate-data/final_annotations.csv", 2),
        DETOX("detox-de.csv","germany", textdetox_multilingual['de'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-de.csv","germany", textdetox_explain['de'], 'Sentence', 'Toxicity Level', 1, 0),

        # ENGLISH: 
        CulturePark("culture-data/English/CONAN/en_data-2.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        #CulturePark("data/English/CONAN/en_data.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/English/CrowS-Pairs-TODO/data.jsonl", "english", "data", "label", "BAD", "GOOD"),
        CulturePark("culture-data/English/EXIST 2021/en_data.jsonl", "english", "data", "label", "BAD", "GOOD"),
        #CulturePark("data/English/EXIST 2021/task1.jsonl", "english", "data", "label", "1", "0"),
        #CulturePark("data/English/HASOC2020/data_finegrained.jsonl", "english", "data", "label"),
        CulturePark("culture-data/English/HASOC2020/data.jsonl", "english", "data", "label", "HOF", "NOT"),
        #CulturePark("data/English/hate-speech-and-offensive-language/data.jsonl", "english", "data", "label"),
        #CulturePark("data/English/HateEval 2019/data.jsonl", "english", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/English/HateEval 2019/hateval2019_en_test.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/English/MLMA hate speech/data-2.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        #CulturePark("data/English/MLMA hate speech/directness.jsonl", "english", "data", "label", "1", "0"),
        #CulturePark("data/English/OLID/data.jsonl", "english", "data", "label", "OFF", "NOT"),
        #CulturePark("data/English/OLID/offense_target.jsonl", "english", "data", "label"),
        #CulturePark("data/English/OLID/offense_classify.jsonl", "english", "data", "label"),
        CulturePark("culture-data/English/OLID/offense.jsonl", "english", "data", "label", "OFF", "NOT"),
        #CulturePark("data/English/SOLID/test_a_tweets_easy.jsonl", "english", "data", "label"),
        #CulturePark("data/English/Toxic Comment Classification Challenge/threat.jsonl", "english", "data", "label", "1", "0"),
        CulturePark("culture-data/English/Toxic Comment Classification Challenge/toxic.jsonl", "english", "data", "label", "1", "0"),
        CulturePark("culture-data/English/hate-speech-and-offensive-language/data.jsonl", "english", "data", "label", {"0", "1"}, "2"),
        MultiHate("multihate-data/captions/en.csv","english", "multihate-data/final_annotations.csv", 1),
        DETOX("detox-en.csv","english", textdetox_multilingual['en'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-en.csv","english", textdetox_explain['en'], 'Sentence', 'Toxicity Level', 1, 0),
        

        # CHINA
        CulturePark("culture-data/China/CDial-Bias/gender-2.jsonl", "china", "data", "label", "1", "0"),
        #CulturePark("culture-data/China/Chinese-Camouflage-Spam-dataset/data-2.jsonl", "china", "data", "label", "Spam", "Ham"),
        MultiHate("multihate-data/captions/zh.csv","china", "multihate-data/final_annotations.csv", 4),
        DETOX("detox-zh.csv","china", textdetox_multilingual['zh'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-zh.csv","china", textdetox_explain['zh'], 'Sentence', 'Toxicity Level', 1, 0),



        #CulturePark("data/China/CValues/cvalues_responsibility_mc.jsonl", "china"),
        #CulturePark("data/China/CValues/output_context_chatgpt.jsonl", "china"),
        #CulturePark("data/China/CValues/output_context_china.jsonl", "china"),

        # BENGALI
        CulturePark("culture-data/Bengali/BAD-Bangla-Aggressive-Text-Dataset/data-2.jsonl", "bengali", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Bengali/Trac2-Task1-Aggresion/aggression-data-2.jsonl", "bengali", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Bengali/Bangla-Abusive-Comment-Dataset/racism.jsonl", "bengali", "data", "label", "1", "0"),
        CulturePark("culture-data/Bengali/Bangla-Abusive-Comment-Dataset/threat.jsonl", "bengali", "data", "label", "1", "0"),
        CulturePark("culture-data/Bengali/Trac2-Task2-Misogynistic/Misogynistic-data-2.jsonl", "bengali", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Bengali/Bengali hate speech dataset/religion_data-2.jsonl", "bengali", "data", "label", "HS", "NOT_HS"),

        # HINDI
        DETOX("detox-hi.csv","hindi", textdetox_multilingual['hi'], 'text', 'toxic', 1, 0),
        MultiHate("multihate-data/captions/hi.csv","hindi", "multihate-data/final_annotations.csv", 5),
        DETOX_EXPLAIN("detoxexplain-hi.csv","hindi", textdetox_explain['hi'], 'Sentence', 'Toxicity Level', 1, 0),
        
        
        # ITALY
        DETOX("detox-it.csv","italy", textdetox_multilingual['it'], 'text', 'toxic', 1, 0),

        # FRANCE
        DETOX("detox-fr.csv","france", textdetox_multilingual['fr'], 'text', 'toxic', 1, 0),

        # ARABIC
        # #CulturePark("data/Arabic/MP/hateSpeech.jsonl", "arabic", "comment", "label", "HS", "NOT_HS"),
        #CulturePark("data/Arabic/MP/offens.jsonl", "arabic", "comment", "label", "OFF", "NOT"),
        # #CulturePark("data/Arabic/MP/VulgarSpeech.jsonl", "arabic", "comment", "label")
        #CulturePark("data/Arabic/OffensEval2020/OffensEval.jsonl", "arabic", "tweet", "label", "OFF", "NOT"),
        # #CulturePark("data/Arabic/OSACT4/dev_data_offens.jsonl", "arabic", "tweet", "label", "OFF", "NOT"),
        #CulturePark("data/Arabic/OSACT4/dev_data.jsonl", "arabic", "tweet", "label", "HS", "NOT_HS"),
        # #CulturePark("data/Arabic/OSACT5/hate_Finegrained.jsonl", "arabic", "tweet", "label", "HS", "NOT_HS"),
        # #CulturePark("data/Arabic/OSACT5/hateSpeech.jsonl", "arabic", "tweet", "label", "HS", "NOT_HS"),
        #CulturePark("data/Arabic/OSACT5/offens.jsonl", "arabic", "tweet", "label", "OFF", "NOT"),
        #CulturePark("data/Arabic/SpamDetect/span_detect_2.jsonl", "arabic", "tweet", "label", "Spam", "Ham"),

    ]

    complete_data = []
    seen_texts = set()

    for source in SOURCES:
        for x in source.read_data():
            if x[0] in seen_texts:
                continue

            complete_data.append({'language': source.get_language(), 'source': source.get_path(), 'text': x[0], 'label': x[1]})
            seen_texts.add(x[0])

    statistics(complete_data)

    with open("annotations.json", "w") as f:
        json.dump(complete_data, f, indent=4)
