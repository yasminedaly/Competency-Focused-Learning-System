from flask import Flask, jsonify
from TopicGenerator import TopicGenerator
import requests

app = Flask(__name__)


@app.route("/get_subjects")
def topic_modelling():
    # Make the GET request to retrieve the data
    response = requests.get('http://localhost/api/data')
    data = response.json()

    topic_generator = TopicGenerator()
    top_words_dict = topic_generator.generate_topics_csv(data)

    # Convert the results to a JSON response
    response_data = jsonify(top_words_dict)

    # Send the JSON response as a POST request
    post_response = requests.post('http://localhost/api/results', json=response_data)

    return "Results sent successfully"


if __name__ == "__main__":
    app.run()
