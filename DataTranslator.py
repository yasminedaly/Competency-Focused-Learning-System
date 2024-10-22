import spacy
from spacy_langdetect import LanguageDetector
from googletrans import Translator
from spacy.language import Language


class DataTranslator:
    def __init__(self):
        self.nlp = None

    def load_model(self, model_name):
        self.nlp = spacy.load(model_name)

    def get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def spacy_language_detection(self, text):
        if not self.nlp:
            raise ValueError("Spacy model not loaded. Call load_model() before language detection.")

        pipeline = list(dict(self.nlp.pipeline).keys())

        if not "language_detector" in pipeline:
            Language.factory("language_detector", func=self.get_lang_detector)
            self.nlp.add_pipe('language_detector', last=True)

        doc = self.nlp(text)
        return doc._.language

    def detect_languages(self, texts):
        if not self.nlp:
            raise ValueError("Spacy model not loaded. Call load_model() before language detection.")

        return [self.spacy_language_detection(text) for text in texts]

    # Define a function to translate English to French using Google Translate API
    def translate_french_to_english(text):
        translator = Translator()
        return translator.translate(text, src='fr', dest='en').text

# # Example usage:
# translator = DataTranslation()
# translator.load_model("en_core_web_sm")
#
# # Apply the language detection to the "text" column in the DataFrame
# df['lang'] = translator.detect_languages(df['all_activities'])
# df['lang'] = df['lang'].apply(lambda x: x['language'])

# Select only the rows where the value in the "lang" column is "en"
# fr_rows = df['lang'] == 'fr'

# Apply the translation function to the "all_activities" column in those rows
# df.loc[fr_rows, 'all_activities'] = df.loc[fr_rows, 'all_activities'].apply(translate_french_to_english)
