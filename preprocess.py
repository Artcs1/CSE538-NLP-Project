import json
import pandas as pd

from abc import ABC, abstractmethod
from datasets import load_dataset

class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def read_data(self):
        """
        Read data from the source.

        Returns:
            list of tuple: Each tuple contains (text, label).
        """
        pass

    @abstractmethod
    def get_language(self):
        """
        Get the language of respective data source.

        Returns:
            string: Contains the language name of the data source.
        """
        pass

    @abstractmethod
    def get_source(self):
        """
        Get the source of respective data source.

        Returns:
            string: Contains the source name or path of the data source.
        """
        pass


class CulturePark(DataSource):
    def __init__(self, path, language, text_key, label_key, label_true, label_false):
        self.path = path
        self.language = language
        self.text_key = text_key
        self.label_key = label_key
        self.label_true = label_true
        self.label_false = label_false

    def read_data(self):
        with open(self.path, "rt") as f:
            data = [json.loads(line) for line in f]

            # Some data have multiple label for true or false
            if isinstance(self.label_true, set) and isinstance(self.label_false, set):
                data = [(
                        x[self.text_key],
                        1 if x[self.label_key] in self.label_true else (
                                0 if x[self.label_key] in self.label_false else -1
                        )
                    ) for x in data if isinstance(x[self.text_key], str)]
                
            elif isinstance(self.label_true, set):
                data = [(
                        x[self.text_key],
                        1 if x[self.label_key] in self.label_true else (
                                0 if x[self.label_key] == self.label_false else -1
                        )
                    ) for x in data if isinstance(x[self.text_key], str)]
                
            elif isinstance(self.label_false, set):
                data = [(
                        x[self.text_key],
                        1 if x[self.label_key] == self.label_true else (
                                0 if x[self.label_key] in self.label_false else -1
                        )
                    ) for x in data if isinstance(x[self.text_key], str)]
                
            else:
                data = [(
                        x[self.text_key],
                        1 if x[self.label_key] == self.label_true else (
                                0 if x[self.label_key] == self.label_false else -1
                        )
                    ) for x in data if isinstance(x[self.text_key], str)]

        return data
    
    def get_language(self):
        return self.language
    
    def get_source(self):
        return self.path


class MultiHate(DataSource):
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

        return data

    def get_language(self):
        return self.language
    
    def get_source(self):
        return self.path


class DETOX(DataSource):
    def __init__(self, path, language, data, text_key, label_key, label_true, label_false):
        self.path = path 
        self.language = language
        self.data = data
        self.text_key = text_key
        self.label_key = label_key
        self.label_true = label_true
        self.label_false = label_false

    def read_data(self):
        data = []
        for item in self.data:
            data.append((item[self.text_key], int(item[self.label_key])))

        return data

    def get_language(self):
        return self.language
    
    def get_source(self):
        return self.path


class DETOX_EXPLAIN(DataSource):
    def __init__(self, path, language, data, text_key, label_key, label_true, label_false):
        self.path = path 
        self.language = language
        self.data = data
        self.text_key = text_key
        self.label_key = label_key
        self.label_true = label_true
        self.label_false = label_false

    def read_data(self):
        data = []
        for item in self.data:
            label = 1
            if item[self.label_key] == 'Low':
                label = 0
            data.append((item[self.text_key], label))

        return data

    def get_language(self):
        return self.language
    
    def get_source(self):
        return self.path


def statistics(data):
    """For each language, count number of positive, negative, and combined data."""
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


if __name__ == "__main__":
    textdetox_multilingual = load_dataset("textdetox/multilingual_toxicity_dataset")
    textdetox_explain = load_dataset("textdetox/multilingual_toxicity_explained")

    # Define all data sources
    SOURCES = [
        # SPANISH
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

        # TURKEY
        CulturePark("culture-data/Turkey/ATC/fold_0_test.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/offensDetect-kaggle2/test.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/offenseCorpus/offens.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/OffensEval2020/OffensEval.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Turkey/offenssDetect-kaggle/turkish_tweets_2020.jsonl", "turkey", "tweet", "label", "OFF", "NOT"),

        # KOREAN
        CulturePark("culture-data/Korean/AbuseEval/data-2.jsonl", "korean", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Korean/CADD/data-2.jsonl", "korean", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/Korean/K-MHaS/data-2.jsonl", "korean", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Korean/Korean-Hate-Speech-Detection/data-2.jsonl", "korean", "data", "label", "HS", {"NOT_HS", ""}),
        CulturePark("culture-data/Korean/KoreanHateSpeechdataset/data-2.jsonl", "korean", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Korean/Waseem/data-2.jsonl", "korean", "data", "label", "OFF", "NOT"),

        # GERMANY
        CulturePark("culture-data/Germany/GermEval/germeval2018.jsonl", "germany", "tweet", "label", "OFF", "NOT"),
        CulturePark("culture-data/Germany/HASOC/hate_off_detect.jsonl", "germany", "tweet", "label", "HOF", "NOT"),
        CulturePark("culture-data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_2.jsonl", "germany", "tweet", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/Germany/MHC/hatecheck_cases_final_german.jsonl", "germany", "tweet", "label", "HS", "NOT_HS"),
        MultiHate("multihate-data/captions/de.csv","germany", "multihate-data/final_annotations.csv", 2),
        DETOX("detox-de.csv","germany", textdetox_multilingual['de'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-de.csv","germany", textdetox_explain['de'], 'Sentence', 'Toxicity Level', 1, 0),

        # ENGLISH
        CulturePark("culture-data/English/CONAN/en_data-2.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/English/CrowS-Pairs-TODO/data.jsonl", "english", "data", "label", "BAD", "GOOD"),
        CulturePark("culture-data/English/EXIST 2021/en_data.jsonl", "english", "data", "label", "BAD", "GOOD"),
        CulturePark("culture-data/English/HASOC2020/data.jsonl", "english", "data", "label", "HOF", "NOT"),
        CulturePark("culture-data/English/HateEval 2019/hateval2019_en_test.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/English/MLMA hate speech/data-2.jsonl", "english", "data", "label", "HS", "NOT_HS"),
        CulturePark("culture-data/English/OLID/offense.jsonl", "english", "data", "label", "OFF", "NOT"),
        CulturePark("culture-data/English/Toxic Comment Classification Challenge/toxic.jsonl", "english", "data", "label", "1", "0"),
        CulturePark("culture-data/English/hate-speech-and-offensive-language/data.jsonl", "english", "data", "label", {"0", "1"}, "2"),
        MultiHate("multihate-data/captions/en.csv","english", "multihate-data/final_annotations.csv", 1),
        DETOX("detox-en.csv","english", textdetox_multilingual['en'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-en.csv","english", textdetox_explain['en'], 'Sentence', 'Toxicity Level', 1, 0),

        # CHINA
        CulturePark("culture-data/China/CDial-Bias/gender-2.jsonl", "china", "data", "label", "1", "0"),
        MultiHate("multihate-data/captions/zh.csv","china", "multihate-data/final_annotations.csv", 4),
        DETOX("detox-zh.csv","china", textdetox_multilingual['zh'], 'text', 'toxic', 1, 0),
        DETOX_EXPLAIN("detoxexplain-zh.csv","china", textdetox_explain['zh'], 'Sentence', 'Toxicity Level', 1, 0),

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
    ]

    # Convert data to JSON format with keys 'language', 'source', 'text', and 'label'
    complete_data = []
    seen_texts = set()

    for source in SOURCES:
        for x in source.read_data():
            # Skip duplicates
            if x[0] in seen_texts:
                continue

            complete_data.append({
                'language': source.get_language(),
                'source': source.get_source(),
                'text': x[0],
                'label': x[1]
            })
            seen_texts.add(x[0])

    # Print statistics
    statistics(complete_data)

    # Export final data
    with open("annotations.json", "w") as f:
        json.dump(complete_data, f, indent=4)
