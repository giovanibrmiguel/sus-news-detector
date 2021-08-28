import streamlit as st
import pickle
from model import *
import time

path = './serialized_model.pkl'
file = open(path, 'rb')
model_predictor = pickle.load(file)
#file.close

def prediction(url_noticia):
    model_predictor.data = url_noticia
    try:
        model_predictor.data = model_predictor.news_please()
    except:
        response = 'Essa URL é inválida! Você deve inserir no campo apenas URLs de notícias relacionadas à pandemia de COVID-19.'
        return response
    model_predictor.data = model_predictor.vec_news()

    response = model_predictor.predict()

    if response == 1:
        response = 'Essa notícia parece ser confiável.'
    elif response == -1:
        response = 'Essa notícia parece ser suspeita. Cheque a fonte!'
    return response

def main():
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Suspicious News Detector</h1> 
    </div> 
    """
      
    st.markdown(html_temp, unsafe_allow_html = True) 

    noticia = st.text_input("Esse é um modelo de machine learning que busca identificar notícias suspeitas sobre COVID-19.")
    if not noticia:
        st.warning('Por favor, cole a URL da sua notícia e aperte enter.')
        st.stop()

    if st.button("Essa notícia é confiável?"): 
        result = prediction(noticia)
        with st.spinner('Estamos analisando sua notícia...'):
            time.sleep(2)
            st.success(result)

if __name__=='__main__': 
    main()