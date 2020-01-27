import numpy as np
import flask
import io
from chatbot import loadChatbot,predictStringInput


app = flask.Flask(__name__)

chatbot_model = loadChatbot('chatbot.h5')

@app.route('/')
def home_endpoint():
    return 'Tryout'

@app.route('/predict', methods=['GET','POST'])
def get_prediction():
    # Works only for a single sample
    if flask.request.method == 'POST':
        data = flask.request.json  # Get data posted as a json
        if data == None:
            data = flask.request.args
        input = data.get('data')
        prediction = predictStringInput(chatbot_model,input)  # runs globally loaded model on the data
    return prediction


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
