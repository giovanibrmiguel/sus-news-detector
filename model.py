# Importing libs
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import pickle
import nltk
#nltk.download('stopwords')  
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from newsplease import NewsPlease



class predictor():

    def __init__(self,regressor):
        self.regressor = regressor
        self.data = None
    
    def predict(self):
        response = self.regressor.predict(self.data)
        return response
    
    def news_please(self):
        news = NewsPlease.from_url(self.data)
        news_text = news.maintext.replace('\n',' ')
        return news_text

    def vec_news(self):
        vocabulario = pd.read_csv("./data/vocabulario.csv").set_index('Unnamed: 0').iloc[:,0].tolist()
        stop_words = set(stopwords.words('portuguese'))
        tokenizer = RegexpTokenizer(r'\w+')

        token_1 = tokenizer.tokenize(self.data)
        token_2 = [word.lower() for word in token_1 if word not in stop_words]
        text_join = " ".join(token_2)
        tokenized_news_join = list()
        tokenized_news_join.append(text_join)

        count_vec = CountVectorizer(vocabulary= vocabulario)
        count_vec_news = count_vec.fit_transform(tokenized_news_join)

        news_df = pd.DataFrame(columns = vocabulario)
        news_list = list()
        values = count_vec_news[0].toarray()[0]
        zipped = zip(vocabulario, values)
        df_dic = dict(zipped)
    
        news_list.append(df_dic)
        noticia_tratada = news_df.append(news_list)
        
        return noticia_tratada

def model_build():
    # Opening training file
    X_train = pd.read_csv("./data/X_train.csv").set_index('Unnamed: 0')
    # Running model
    model = OneClassSVM(kernel = "rbf",
                        gamma= 0.001,
                        nu = 0.4,
                        verbose = False)
    model.fit(X_train)
    model = predictor(model)

    with open('serialized_model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    pickle_file.close()
    
    return model

def main():
    model = model_build()
    # Testing predict function
    #noticia_tratada =  pd.read_csv("noticia_tratada.csv").set_index('Unnamed: 0')
    model.data = 'https://g1.globo.com/bemestar/coronavirus/noticia/2021/05/19/apos-8-dias-em-queda-curva-de-mortes-por-covid-no-brasil-volta-a-indicar-estabilidade.ghtml'
    model.data = model.news_please()
    model.data = model.vec_news()
    response = model.predict()
    print('globo : ',response)

    model.data = 'https://www.stylourbano.com.br/vacina-com-proteina-magneto-pode-controla-remotamente-o-cerebro-e-o-comportamento/'
    model.data = model.news_please()
    model.data = model.vec_news()
    response = model.predict()
    print('magneto :', response)

if __name__ == '__main__':
    main()