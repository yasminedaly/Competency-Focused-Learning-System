import requests
import json
import pandas as pd
import os


class WikipediaAPI:
    def __init__(self):
        self.url = "https://en.wikipedia.org/w/api.php"
        self.params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": True
        }

    def get_summary(self, title):
        self.params["titles"] = title
        response = requests.get(self.url, params=self.params)
        data = json.loads(response.text)
        try:
            pages = data["query"]["pages"]
            summary = list(pages.values())[0]["extract"]
        except KeyError:
            summary = ""
        return summary


class SpecialtySummaries:
    def __init__(self):
        self.specialties_list = [
            "Software Engineering",
            "Data Science",
            "Artificial Intelligence",
            "Cybersecurity",
            "Network Engineering",
            "Database Management",
            "Web Development",
            "Mobile Application Development",
            "Cloud Computing",
            "Virtual Reality",
            "Augmented Reality",
            "XR",
            "Blockchain",
            "IoT",
            "Robotics",
            "Computer Graphics",
            "Game Development",
            "Information Systems"
        ]
        self.api = WikipediaAPI()

    def get_specialty_summaries(self):
        specialties = pd.DataFrame(columns=["Specialty", "Summary"])
        for specialty in self.specialties_list:
            summary = self.api.get_summary(specialty)
            row = {"Specialty": specialty, "Summary": summary}
            specialties = specialties.append(row, ignore_index=True)
        return specialties

    def group_and_save_data(filename, column_name):
        data = pd.read_csv(filename)
        grouped_data = data.groupby(column_name)

        for name, group in grouped_data:
            filename = '{}.csv'.format(name)
            group.to_csv(filename, index=False)

    def process_csv_files(output_dir):
        # Iterate over each CSV file in the output directory
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.csv'):
                # Get the full path to the CSV file
                file_path = os.path.join(output_dir, file_name)

                # Import the dataset
                df = pd.read_csv(file_path)

                # Delete rows containing specific words in "processed_activities" column
                keywords = ['stage', 'stagiaire', 'trainee', 'intern', 'internship']
                df = df[~df['processed_activities'].str.contains('|'.join(keywords), case=False)]

                # Perform further operations on the modified dataset if needed

                # Save the modified dataset back to the same file
                df.to_csv(file_path, index=False)

# Get the current working directory
# current_dir = os.getcwd()

# Path to the output directory relative to the current directory
# output_dir = os.path.join(current_dir, 'output')

# Call the function to process the CSV files in the output directory
# process_csv_files(output_dir)

# Example usage:
# specialty_summaries = SpecialtySummaries()
# specialties = specialty_summaries.get_specialty_summaries()
# print(specialties)
