import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import json
import glob
import os

from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor


class SkillAnnotator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)

    def annotate_with_error_handling(self, text):
        try:
            return self.skill_extractor.annotate(text)
        except IndexError:
            return None
        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None

    @staticmethod
    def get_skill_type(skill_id, skilldb):
        if isinstance(skill_id, list):
            return [SkillAnnotator.get_skill_type(id, skilldb) for id in skill_id]
        elif skill_id in skilldb.index:
            return skilldb.loc[skill_id, 'skill_type']
        else:
            return None

    @staticmethod
    def replace_none_with_empty_dict(d):
        if isinstance(d, list):
            for i in range(len(d)):
                if d[i] is None:
                    d[i] = {}  # replace None with an empty dictionary
                elif isinstance(d[i], (dict, list)):
                    SkillAnnotator.replace_none_with_empty_dict(d[i])
        elif isinstance(d, dict):
            for k, v in d.items():
                if v is None:
                    d[k] = {}  # replace None with an empty dictionary
                elif isinstance(v, (dict, list)):
                    SkillAnnotator.replace_none_with_empty_dict(v)

    def process_files(self, grouped_data):
        skilldb = pd.read_json("skill_db_relax_20.json", orient='records')
        skilldb = skilldb.transpose()

        for name, group in grouped_data:
            csv_files = glob.glob(f"{name}*.csv")
            for csv_file in csv_files:
                # Read the CSV file into a pandas dataframe
                file_data = pd.read_csv(csv_file)
                file_data = file_data.astype(str)
                annotations = file_data["processed_activities"].apply(self.annotate_with_error_handling)
                annotations_file = f"annotationstest.json"
                annotations.to_json(annotations_file, orient="records")

                # Load the data from the JSON file
                with open("annotationstest.json", 'r') as f:
                    data = json.load(f)

                # Call the function on your data
                SkillAnnotator.replace_none_with_empty_dict(data)

                # Save the updated data back to the JSON file
                with open("annotationstest.json", 'w') as f:
                    json.dump(data, f)

                annods = pd.read_json("annotationstest.json", orient='records')
                annods['skill_id'] = annods['results'].apply(
                    lambda x: [match['skill_id'] for match in x['full_matches']] + [
                        scored['skill_id'] for scored in x['ngram_scored']] if isinstance(x, dict) and 'full_matches' in x and 'ngram_scored' in x else [])
                annods['skill_name'] = annods['results'].apply(
                    lambda x: [match['doc_node_value'] for match in x['full_matches']] + [
                        scored['doc_node_value'] for scored in x['ngram_scored']] if isinstance(x, dict) and 'full_matches' in x and 'ngram_scored' in x else [])

                # Join the datasets on the common column
                file_data = pd.merge(file_data, annods, left_on='processed_activities', right_on='text')
                file_data = file_data[file_data['skill_id'].apply(lambda x: len(x) > 0)]
                skill_types = file_data['skill_id'].apply(
                    lambda x: SkillAnnotator.get_skill_type(x, skilldb))
                file_data['skill_type'] = skill_types

                # Create empty columns
                file_data['hard_skills'] = ''
                file_data['soft_skills'] = ''
                file_data['certification'] = ''

                # Iterate over the rows
                for index, row in file_data.iterrows():
                    for i in range(len(row['skill_type'])):
                        # Get the corresponding skill name
                        skill_name = row['skill_name'][i]
                        skill_type = row['skill_type'][i]

                        # Assign the skill name to the corresponding column
                        if skill_type == 'Hard Skill':
                            row['hard_skills'] += skill_name + ', '
                        elif skill_type == 'Soft Skill':
                            row['soft_skills'] += skill_name + ', '
                        elif skill_type == 'Certification':
                            row['certification'] += skill_name + ', '

                    # Remove trailing comma and whitespace from each column
                    row['hard_skills'] = row['hard_skills'].strip(', ')
                    row['soft_skills'] = row['soft_skills'].strip(', ')
                    row['certification'] = row['certification'].strip(', ')

                # Remove original skill_name and skill_type columns
                file_data = file_data.drop(['skill_name', 'skill_type', 'skill_id', 'results', 'text'], axis=1)

                # Get the name of the output file
                output_file_name = os.path.splitext(csv_file)[0] + '_output.csv'

                # Save the updated dataframe to a new CSV file
                file_data.to_csv(output_file_name, index=False)

# annotator = SkillAnnotator()
# annotator.process_files(grouped_data)
