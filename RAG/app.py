import streamlit as st 
from RAG import RAG

st.title('RAG Foundation project')


st.header('hello')
st.progress(50,text='Loading')
question = st.text_area('Enter your question here')
click_btn = st.button('Click to answer your questions')

rag = RAG()

if click_btn:
    if question is not '':
        answer = rag.generate_text(question)
        st.text(answer)
    else:
        print('warning')