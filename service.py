from flask import Flask, request 
import numpy as np
import pandas as pd
import pickle
import os
from model import *

# instancia o objeto do Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict(path = './serialized_model.pkl'):
    #params = request.args.to_dict()
    #print(params)
    file = open(path, 'rb')
    model_predictor = pickle.load(file)
    file.close

    noticia = ' a s d a s sa asdas as ada teste AAAAAAAAAAAA 11111111111232323232'
    
    model_predictor.data = noticia
    model_predictor.data = model_predictor.vec_news()
    
    response = model_predictor.predict()

    if response == 1:
        response = 'Essa notícia parece ser confiável.'
    elif response == -1:
        response = 'Essa notícia parece ser suspeita. Cheque a fonte!'
    return response
    #print(model_predictor)
    #response = model_predictor.predict()
    #print(response)    

if __name__ == '__main__':
    #print(os.getenv('USERNAME'))
    #if (os.getenv('USER') == 'eti') or (os.getenv('USERNAME') == 'eduar'):
    #    app.run(host='0.0.0.0', port=8080, debug=True)
    #else:
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port)
