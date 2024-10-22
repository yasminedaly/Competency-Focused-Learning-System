import locale
import pandas as pd
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import contractions
import re


class Preprocessor:
    def __init__(self):
        locale.getpreferredencoding = lambda: "UTF-8"
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.stop_words = set(stopwords.words('english') + list(punctuation))
        self.lemmatizer = WordNetLemmatizer()
        self.english_stop_words = set(stopwords.words('english'))
        self.custom_stop_words = {'a',',','.',':','!','?',';','&','^','``','@','°','(',')','=','_','-','{','}','#','~',
                                  '/','*','-','+','<','>','²','|','about', 'above', 'across', 'after', 'afterwards',
                                  'b.','c.','1','2','3','4','5','6','7','8','9','janvier','fevrier','mars','avril',
                                  'mai','juin','juillet','aout','septembre','octobre','novembre','decembre','','le',
                                  'the','il','elle','to','à','de','du', 'all', 'almost', 'alone', 'along', 'already',
                                  'also', 'although', 'always', 'am', 'among', 'amongst', 'amongst', 'amount', 'an',
                                  'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are',
                                  'around', 'as', 'at', 'back', 'be', 'became', 'become', 'becomes', 'becoming', 'been',
                                  'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between',
                                  'beyond', 'bill', 'both', 'bottom', 'by', 'call', 'co', 'con', 'could', 'couldnt',
                                  'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each',
                                  'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'etc', 'even',
                                  'ever', 'every', 'everyone', 'everything', 'everywhere', 'few', 'fifteen', 'fify',
                                  'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty',
                                  'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
                                  'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby',
                                  'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however',
                                  'hundred', 'i', 'ie', 'if', 'in','connue','inc', 'indeed', 'into', 'is', 'it', 'its',
                                  'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made',
                                  'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
                                  'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither',
                                  'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'nobody', 'none', 'noone',
                                  'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one',
                                  'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves',
                                  'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
                                  'same', 'see','.','..','...','in','out','a', 'afin', 'ai', 'ainsi', 'après',
                                  'attendu', 'au', 'aucun', 'aucune', 'aujourd', "aujourd'hui", 'auquel', 'aussi',
                                  'autre', 'autres', 'aux', 'avant', 'avec', 'avoir', 'c', 'car', 'ce', 'ceci', 'cela',
                                  'celle', 'celles', 'celui', 'cependant', 'certain', 'certaine', 'certains', 'ces',
                                  'cet', 'cette', 'ceux', 'chez', 'ci', 'combien', 'comme', 'comment', 'concernant',
                                  'contre', 'd', 'dans', 'de', 'debout', 'dedans', 'dehors', 'delà', 'depuis',
                                  'derrière', 'des', 'désormais', 'desquelles', 'desquels', 'dessous', 'dessus', 'deux',
                                  'devant', 'devers', 'devra', 'divers', 'diverse', 'diverses', 'doit', 'donc', 'dont',
                                  'du', 'duquel', 'durant', 'dès', 'elle', 'elles', 'en', 'entre', 'environ', 'est',
                                  'et', 'etc', 'eu', 'eux', 'excepté', 'hormis', 'hors', 'hélas', 'hui', 'il', 'ils',
                                  'j', 'je', 'jusqu', 'jusque', 'l', 'la', 'laquelle', 'le', 'lequel', 'les',
                                  'lesquelles', 'lesquels', 'leur', 'leurs', 'lors', 'lui', 'là', 'ma', 'mais', 'malgré',
                                  'me', 'merci', 'mes', 'mien', 'mienne', 'miennes', 'miens', 'moi', 'moins', 'mon',
                                  'moyennant', 'même', 'mêmes', 'n', 'ne', 'ni', 'nommés', 'non', 'nos', 'notre', 'nous',
                                  'nouveau', 'nouveaux', 'nul', 'nulle', 'o', 'où', 'ont', 'ou', 'outre', 'par', 'parmi',
                                  'partant', 'pas', 'passé', 'pendant', 'plein', 'plus', 'plusieurs', 'pour', 'pourquoi',
                                  'proche', 'près', 'puis', 'qu', 'quand', 'que', 'quel', 'quelle', 'quelles', 'quels',
                                  'qui', 'quoi', 'quoique', 'revoici', 'revoilà', 's', 'sa', 'sans', 'sauf', 'se',
                                  'selon', 'seront', 'ses', 'si', 'sien', 'sienne', 'siennes', 'siens', 'sinon', 'soi',
                                  'soit', 'son', 'sont', 'sous', 'suivant', "ta", "tandis", "te", "tel", "telle",
                                  "telles", "tels", "tes", "toi", "ton", "tous", "tout", "toute", "toutes", "tu", "un",
                                  "une", "va", "vers", 'so', 'some'}
        self.english_stop_words = self.english_stop_words.union(self.custom_stop_words)

    def expand_contractions(self, text):
        expanded_words = []
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        expanded_text = ' '.join(expanded_words)
        return expanded_text

    def preprocess_text(self, text):
        text = self.expand_contractions(text)
        text = ' '.join([word for word in word_tokenize(text) if word.lower() not in self.english_stop_words])
        text = re.sub(r'[^a-zA-ZéèêëîïôöûùüçàâæœÉÈÊËÎÏÔÖÛÙÜÇÀÂÆŒ]+', ' ', text)
        return text

    def preprocess_dataframe(self, df, column_name):
        df[column_name] = df[column_name].apply(self.preprocess_text)
        return df


# Example usage:
# df = pd.read_csv("english_dataset.csv")
# preprocessor = Preprocessor()
# df = preprocessor.preprocess_dataframe(df, "all_activities")
