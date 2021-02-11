import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_cors import CORS
import pickle
import time
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json 
import pickle
app = Flask(__name__)
#CORS(app)
vocab_size = 12708
maxlen = 30
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)


#model = pickle.load(open('model.pkl', 'rb'))
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST']) #'GET',
def predict():
    '''
    For rendering results on HTML GUI
    '''
    temp=request.get_data(as_text=True)
    start = time.time()
    new_string = temp.replace("+", " ")
    new_string = new_string.replace("%2C", " ")
    new_string = new_string.replace("%3B", " ")
    new_string = new_string.replace("text_input=", " ")
    new_string= " ".join(new_string.split())
    #print(temp)
    new=[new_string]
    tok = tokenizer1.texts_to_sequences(new)
    pad = pad_sequences(tok, padding='post', maxlen=maxlen)
    y_pred1 = loaded_model.predict(pad)
    end = time.time()
    pred = np.argmax(y_pred1, axis=-1)
    if pred == 1:
        output = "Bully"
    else:
        output = "Non Bully"

    tt = end - start
    cc = "all bal chal"
    return render_template('index.html', Input_text1='Input Sentence : {}'.format(new_string),
                           prediction_text='Input Sentence Indicates : {}'.format(output)
                           , Time_taken='Time Taken for Prediction : {:.2f}'.format(tt),
                           nonbully_prob ='Non-Bully  Probability : {:.2f}'.format(y_pred1[0,0]),
                           bully_prob='Bully Probability : {:.2f}'.format(y_pred1[0,1]))


if __name__ == "__main__":
    app.run(debug=True)