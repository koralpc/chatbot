# chatbot
AWS(EC2) deployed Chatbot application


Hello there! Today, we will be building a simple chatbot model from scratch and deploy it in an EC2 instance to be able to do remote calls to our trained model.

Since all steps are done by ourselves, this tutorial might take quite some portion of your time, so I would say buckle up from now on, and set aside a block of your time for this.

## Table Of Contents

* [Building chatbot with Keras](#keras)
* [Implementing web functionality with Flask](#flask)
* [Setting up Amazon EC2 instance](#ec2)
* [Containerizing application via Docker](#docker)
* [Using remote call functionality in HTML/JQuery](#html./jquery)
* [Conclusion](#conclusion)



## Building Chatbot with Keras <a name ="keras">

### Setting up the dataset

In creating our chatbot, we will need a dataset prior to training. Therefore, we will have a
json document, in which the structure will look as this:

```json

{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you ?", "Is anyone there ?", "Hello", "Good day", "What's up ?" , "How is your day ?"],
         "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["See you", "See you later", "Goodbye", "I am Leaving", "Have a Good day","Bye"],
         "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!"],
         "context_set": ""
        },
        {"tag": "help",
         "patterns": ["Can you help me ?", "What is the topic ?", "What is this post about ?"],
         "responses": ["You can find the info by reading the whole post" , "You can contact Koralp for more info"],
         "context_set": ""
        },
        {"tag": "name",
         "patterns": ["what is your name", "what should I call you", "whats your name?"],
         "responses": ["You can call me Alfred.", "I'm Alfred!"],
         "context_set": ""
        },
   ]
}

```
Here, there are three elements that we are interested in:
* **Tag** : The class label/Denotes which category the sentences fall in. For example, "Hi","How are you", etc. fall into the _greeting_ category
* **Patterns** : These denote sample user inputs. The more we have of these, the better the coverage of our chatbot will be. For demonstration case, couple of simple sentences for each tag will suffice
* **Responses** : As the name gives it away, these sentences will be the responses selected after our model makes its prediction about which _tag_ the response will come from

### Parsing & Vocabulary construction

Second step to preparing the dataset is to parse it, and construct a corpus vocabulary that is essential for language processing tasks.

We start by importing the necessary libraries that will aid us throughout the projects

```python
#Import modules
import io

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()

import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

import random

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

```

The functionalities for the imported libraries will be explained later on through the project.

After importing the libraries, we start with parsing json data, and tokenizing the strings into words in order to add them to the dictionary:

```python
#Tokenization of patterns
def tokenize(data,stemmer):
    flat_json_df = json_normalize(data['intents'])

    all_words = []
    for pattern in flat_json_df['patterns']:
        for pat in pattern:
            all_words.extend(nltk.word_tokenize(pat))

    all_words = [lancaster_stemmer.stem(w.lower()) for w in all_words]
    all_words = set(all_words)
    return all_words,flat_json_df
```

First line, `json_normalize` takes the json file we have, and converts it into a _pandas_ `DataFrame`.

We use this dataframe, and for each **pattern** array corresponding to a given **tag** we tokenize its contents, and then pass the individual word collection through the **stemmer** we imported earlier. Stemmer, LancasterStemmer specifically in this example, is an object that reduces words into their roots. For example, when using stemmer, the words become

---
Hi -> Hi

Running -> Run

---

Therefore, some words are already at root, thus unchanged, whereas others are reduced to their roots. This step is of importance in our application, because it helps to build a more **robust** text processing scheme in case of **high variability** in user input

Using the parsed words, we now create our vocabulary with the helper function

```python
# Create vocabulary from the corpus
def create_vocabulary(corpus):
    vocabulary_dict = {}
    NB_OF_WORDS = len(corpus)
    for idx,word in enumerate(corpus):
        vocabulary_dict[word] = idx+2
    vocabulary_dict['unk'] = 1
    return vocabulary_dict
```

This is a simple dictionary creation step. However, there are two key factors we need to pay attention to:

* Value of 0 : In our application, we will use 0 values to pad sequences upto a constant length, s.t. they can be fed into the network. Therefore, we will be masking _zero_ inputs, hence our vocabulary should not include those.
* Value of _unkown_ words : Another problem that can occur is that, our vocabulary is currently very small and limited. Then, if a user inputs a sentence that includes words that are not in our vocabulary, our application can become erroneous. In order to prevent that, we encode the key `unk` as value of 1, in which the model will interpret as a _unkown_ word.

Since the vocabulary is ready now, we will use it to encode our original training dataset.
We do this via another helper function

```python
def numerate_text(vocabulary,text):
    numerized_text = []
    for t in text:
        output_vec = []
        token = nltk.word_tokenize(t)
        for inner_t in token:
            stemmed = lancaster_stemmer.stem(inner_t.lower())
            if stemmed in vocabulary.keys():
                output_vec.append(vocabulary[stemmed])
            else:
                output_vec.append(vocabulary['unk'])
        numerized_text.append(output_vec)
    return numerized_text

def numerate_string(vocabulary,str):
    numerized_str = []
    token = nltk.word_tokenize(str)
    for inner_t in token:
        stemmed = lancaster_stemmer.stem(inner_t.lower())
        if stemmed in vocabulary.keys():
            numerized_str.append(vocabulary[stemmed])
        else:
            numerized_str.append(vocabulary['unk'])
    numerized_str = [numerized_str]
    return numerized_str
```

Here, `numerate_text` takes an input array of sentences, first tokenizes and stems them, so that they are similar to the stored words in our vocabulary. Then, for the matching key in our dictionary, it returns the number value. Hence, the output result for this method is a number based representation of our sentences.

For example,

---

'Hi there' -> [3,8]

'How are you' -> [5 2 6]

---

`numerate_string` does the exact functionality, but it works on a single sentence, rather than an array of sentences

There is one more step left in the _Parsing_ part, that is converting class tags into **one-hot encoded** vectors.

Currently, our class tags are _strings_, and not applicable for training. Therefore, by converting the labels into one-hot vectors, we will be able to use a multi-neuron output at our model, and give a probabilistic estimation of the predicted class. Then, the predicted number will be re-converted to the actual class tag to select a response from it
We will again use a helper function to do that

```python
def prepare_labels(dataframe):

    class_dict = {}
    #One-hot encode output
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(np.asarray(dataframe['tag']).reshape(-1,1))

    all_labels = []
    for i in dataframe.index:
        for l in range(len(dataframe.iloc[i]['patterns'])):
            all_labels.append(onehot_encoded[i])
        class_dict[np.argwhere(onehot_encoded[i] == 1)[0][0]] =  dataframe.iloc[i]['tag']

    return np.array(all_labels,dtype = 'float32'),class_dict

```

This function returns the one-hot encoded labels, and a dictionary that stores the mapping between original class tags and their one-hot vector values and concludes our parsing step. Now, we can train our model.


### Training the Neural Network

For this task, we will be using a simple LSTM network with a Feed Forward layer connection at output. The reason we choose this network is that, it is a quite powerful network, so even with larger datasets, it can easily scale up while maintaining high accuracy, so this code is more reproduceable.Moreover, an Embedding layer is of high importance here, because it can help us to still perform well when a user inputs unknown words to our vocabulary. Therefore, using these layers saves us from the trouble of hardcoding all question-answer pairs, and allows a much more scalable system. For the network, we will be using Keras API, because of its simplicity, and Tensorflow support, which can be beneficial for custom network layers. Hence, the code for the network is :

```python
def trainChatNN(X,y,vocabulary_length):
    X_reshaped = X.reshape(X.shape[0],1,X.shape[1])

    input_dim = X_reshaped.shape[1:3]
    output_classes = y.shape[1]
    ##TRAINING NETWORK NOW##
    model = Sequential()

    model.add(Input(input_dim[1]))
    model.add(Embedding(vocabulary_length + 1,input_dim[1]//2,input_length = 1,mask_zero =  True))
    model.add(LSTM(input_dim[1]*3, activation = 'relu' , return_sequences = False))
    model.add(Dense(input_dim[1]*2, activation = 'relu'))
    model.add(Dense(output_classes,activation ='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3000)

    history = model.fit(X,y, epochs=3000,verbose = 1,callbacks = [early_stop])
    model.save('chatbot.h5')
    return model


def loadChatbot(filename):
    loaded_model = tf.keras.models.load_model(filename)
    return loaded_model


def predictStringInput(model,str):
        numerical = numerate_string(vocab,str)
        padded = pad_sequences(numerical, maxlen=MAX_SEQ_LENGTH)
        padded = padded.reshape(1,padded.size)
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)
        predicted_tag = label_dict[predicted_class]
        print('Input is: {0} - Predicted tag: {1} - Confidence : {2:.2f}'.format(str,predicted_tag, np.max(prediction)))
        responses = json_df['responses'][json_df['tag'] == predicted_tag]
        total_responses = len(responses.iloc[0])
        randidx = np.random.randint(0,total_responses)
        print('Response is: {0}'.format(responses.iloc[0][randidx]))
        return responses.iloc[0][randidx]
```

All of the method names are descriptive of their functionality, and will be of importance when we start building our Flask application. Now, one last step is remaining in this section, which is putting all these helper functions together.

### Finalizing the class and training

In order to finalize the implementation, we follow the code below.

```python
#Load data
with open('intents.json') as file:
        data = json.load(file)

#Tokenize all words and apply stemming operation
all_words,json_df = tokenize(data,lancaster_stemmer)

#Gather all tags
tags = sorted(json_df['tag'])

#Create vocabulary from stemmed words(Not an embedding, just a vocabulary!)
vocab = create_vocabulary(all_words)

#Convert character arrays to numerical values, w.r.t our vocabulary
numerized = json_df['patterns'].apply(lambda x : numerate_text(vocab,x))

#Create the target label dataset
labels,label_dict = prepare_labels(json_df)

#Pad sequences to MAX_SEQ_LENGTH
padded = []
for n in numerized:
    padded.extend(list(pad_sequences(n, maxlen=MAX_SEQ_LENGTH)))
padded = np.array(padded,dtype = 'float32')

chatbot = trainChatNN(padded,labels,len(vocab))
```

Hence, when we run this code, we can see in the console the training epochs of our LSTM model, and the trained model will be saved to the working directory under the name of `chatbot.h5`. Do not delete this model, as we will be using it in our web application. However, once trained, comment the last line, so when Flask app is importing this class, the model will not be trained again.

## Implementing web functionality with Flask <a name ="flask">

Second section in our project is to build the Flask application. Flask is a quick and easy web application framework, that enables us to deploy Python functionalities in a webhost using RESTful API.

Here, I will first print the code for the Flask app, as it is not much, and then explain the details. So, the code for the Flask app is:

```python
import numpy as np
import flask
import io
from chatbot import loadChatbot,predictStringInput


app = flask.Flask(__name__)

chatbot_model = loadChatbot('chatbot.h5')

@app.route('/')
def home_endpoint():
    return 'Tryout'

@app.route('/predict', methods=['POST'])
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
```

Here, we first have our import statements, which are trivial. However, pay attention that we have imported two functions `loadChatbot` and `predictStringInput` from our **chatbot** class we just have implemented. The reason is, we will be using these methods in one our our flask calls.

`app = flask.Flask(__name__)` creates the _Flask_ object. Here, `loadChatbot` loads the **chatbot.h5** file, which is the trained LSTM model, from our directory. Hence, once training our network, we can simply use its functionality over and over again without needing to run the training process again.

`@app.route('/predict', methods=['POST'])` is a routing command, which specifies the functionality for the url extension _/predict_  and which HTTP methods it is accepting. Here, we will use this url to predict the output of an input string from the user. The method `get_prediction` defined under this line extracts the string that is entered by the user as input, and returns an **HTTP 201** response with the response of the chatbot in the response body.

We can test if our Flask application is working properly via a **cURL** request. cURL is a cmd functionality that enables us to send HTTP requests to a URL and offers many other functionalities. First we run the command `python flaskApp.py(or whatever the name of your file is)` to run our flask app. Next, we open a new cmd window and input

`curl -d {\"data\":\"What's%20up?\"} -H "Content-Type: text/plain" -X POST http://127.0.0.1:5000/predict`

and we should see the response of the chatbot there

![curl output](../assets/images/Post-assets/chatbot-images/curl-output.jpg)

Here, %20 is to encode whitespace, so the curl command can work properly. If you see the results at this step, then congratulations, you have locally deployed your Keras model to a simple web application. Now, the next step is to put this model into a cloud instance, so it is accessible 7/24 and from outside of your local network.

## Setting up Amazon EC2 instance <a name = "ec2">

I will not be talking much about setting up the EC2 instance from stratch, however there is one key point in the settings for this application, when creating the instance, go to the Security tab, and make sure to add another connection setting which uses HTTP.

Now, when the instance is set, you will alse get a key pair for your instance. Save the key, because we will be using it in a moment to connect to instance via ssh from cmd, and also via FileZilla to send files.

### FileZilla client

We will use FileZilla to transfer our local files into our EC2 instance.
Therefore, we first open the Settings tab, and go to SFTP panel. Here, we click the _Add key file_  option, and add the Amazon EC2 key we just stored.

Next, we go to the File -> Site Manager. Here, we select New Site, and we fill in the forms accordingly

Protocol -> SFTP
Host -> Public DNS of your EC2 instance
Username -> ec2-user

Click Connect -> OK and then you shall have access to the file system of your instance. Now we can transfer the Python code, the json file and the saved model to a directory of our choice in the EC2 instance. For this project, I choose to create a directory called Chatbot and transfer files to there.

### SSH connection to the instance

One of the ways to connect to the EC2 instance is via ssh. In order to do that we can use this command below

`ssh -i key-file-path ec2-user@public-dns`

Here, `key-file-path` will be replaced by your key file, and `public-dns` will be replaced by the Publi DNS of your instance. If you got these correctly, you should be seeing the EC2 terminal now!

## Containerizing application via Docker <a name = "docker">

Since we have the connection with EC2 terminal, now its time to containarize our application, and run it. In order to do that, you would need to install Docker, and the official Tensorflow image for Docker. Once these steps are complete, we can create a container via the command

``

This command is important, because first, we open up the port 5000 of the container to outside world.(Which is the port our Flask app uses!) Second, we will be mounting the directory that our Python files belong to this container. Meaning that the changes made in our local device, once reuploaded via FileZilla, will be directly seen in the container.


Now, once we have the container, we can execute the bash/terminal via the command

`docker exec -it container-name bash`

If you did the steps correctly, the output should look as :

![curl output](../assets/images/Post-assets/chatbot-images/tensorflow-docker.jpg)

Now, you can run your _Python_  file that includes the _Flask_  app from here.

To check if the web instance is working correctly, we can use the **curl** requests again, but this time we will send the request to the IP address of the EC2 instance`. The command is:

`curl -d {\"data\":\"Hello\"} -H "Content-Type: text/plain" -X POST http://Public-IP:5000/predict`

If you get a response in your terminal now, Congratulations! you have finished the backend part for this project. Now, the only part remaining is to desing a simple front-end, that can take user input and fetch the prediction from the server.

## Using remote call functionality in HTML/JQuery <a name = "html./jquery">
This section is concerned with the front-end functionalities of the project. Now, since I am not good at this part, we will be doing a simple HTML input form, and a jQuery request to the server, as it is easy to use.

Simply, the code you will need to create this is below.

```HTML
<html>
<h3> Ask the bot : </h3>

<form id="chatbot-in">
  Enter text: <input type="text" id="simpletext"><br>
   <input type="button" id="btn-sbmt" value="Chat">
</form>

<div> <strong> Response : </strong>  <div id="changeable"> </div>  </div>  

<script>

$("#btn-sbmt").click(function(){
var bla = $('#simpletext').val();
  $.ajax({
  url: "http://Public-IP:5000/predict",
  type: 'POST',
  contentType: 'text/plain',
  //dataType: 'text',
  data: '{"data":"'+ bla + '"}',
  success: function (response) {
     //alert(response);
     $("#changeable").text(response);
    //console.log(response)
  },
  error: function(){
    alert("Cannot get data");
  }
});
console.log('Ajax called')
})
</script>
</html>
```

Here, replace the Public-IP field with the IP of your own instance.Then, you are done. Just put this code in your webpage, and the form will be ready to submit with the response from server printed as a changeable text.

## Conclusion <a name = "conclusion">

If you have made it this far, well done! Now, you have a mini full-round project that utilizes many different components of development. As a simple demonstration, I put up the working example in my website [Koralp's page](https://koralpc.github.io), so you can test it in case you were not you were not able to finish the tutorial.

Hope you liked the tutorial! For any questions/comments, leave a message to the comments section below and I will try to respond as ASAP.

You can also find the code in my Github repo

[GitHub](https://github.com/koralpc/chatbot)
