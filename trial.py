import os
import re
import sys
import time
import datetime
import requests
import docx2txt
from glob import glob
import plost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import importlib.util
from io import BytesIO
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle
from pickle import load
from datetime import datetime
from collections import Counter
import hydralit_components as hc
import spacy
from spacy.matcher import Matcher
from time import sleep
import streamlit as st
from streamlit_tags import st_tags
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize

sys.coinit_flags = 0
# load pre-trained model
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the KNN model
mfile = BytesIO(requests.get('https://github.com/Sonal008/Resume-Classification/blob/main/knn_model.pkl?raw=true').content)
model = load(mfile)

mfile1 = BytesIO(requests.get('https://github.com/Sonal008/Resume-Classification/blob/main/tfidf.pkl?raw=true').content)
model1 = load(mfile1)

# Load the preprocessed resume data

word = pd.read_csv('Resume word counts.csv').drop(columns=['Resume'])
clean = pd.read_csv('Resume cleaned.csv')


st.set_page_config(layout='wide',initial_sidebar_state='collapsed')

# specify the primary menu definition
menu_data = [{'icon': "far fa-sticky-note", 'label':"About"},
    {'icon': "far fa-chart-bar", 'label':"Resume Data Analysis"},
    {'icon': "far fa-file-word", 'label':"Resume Classification"},] #no tooltip message]



over_theme = {'txc_inactive': 'white','menu_background':'purple','txc_active':'black','option_active':'white'}
font_fmt = {'font-class':'h2','font-size':'150%'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    login_name=None,
    hide_streamlit_markers=False, 
    sticky_nav=True,
    sticky_mode='pinned', )
 


if menu_id == 'Home':
    st.markdown("""<style>.stProgress .st-bo {color: purple;}</style>""", unsafe_allow_html=True)

    progress = st.progress(0)
    for i in range(100):
        progress.progress(i+1)
        sleep(0.001)
    
    st.markdown("<h1 style='text-align: center; color: black;'>RESUME SCREANING AND CLASSIFICATION</h1>", unsafe_allow_html=True)
    st.image("https://ursusinc.com/wp-content/uploads/2020/09/Resume-Blog-Animation-1080.gif")



#about page
if menu_id == "About":
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1)

    st.markdown("<h1 style='text-align: center; color: black;'>BUSINESS OBJECTIVE </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: justify; font-size:180%; font-style: italic; color: black;'> The document classification solution should significantly reduce the manual human effort in the HRM. It should achieve a higher level of accuracy and automation with minimal human intervention.</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'> ABSTRACT </h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size:140%;'> A resume is a brief summary of your skills and experience. Companies recruiters and HR teams have a tough time scanning thousands of qualified resumes. Spending too many labor hours segregating candidates resume's manually is a waste of a company's time, money, and productivity. Recruiters, therefore, use resume classification in order to streamline the resume and applicant screening process. NLP technology allows recruiters to electronically gather, store, and organize large quantities of resumes. Once acquired, the resume data can be easily searched through and analyzed. Resumes are an ideal example of unstructured data. Since there is no widely accepted resume layout, each resume may have its own style of formatting, different text blocks and different category titles. Building a resume classification and gathering text from it is no easy task as there are so many kinds of layouts of resumes that you could imagine.</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'> INTRODUCTION </h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size:140%; '> In this project we dive into building a Machine learning model for Resume Classification using Python and basic Natural language processing techniques. We would be using Python's libraries to implement various NLP (natural language processing) techniques like tokenization, lemmatization, parts of speech tagging, etc. A resume classification technology needs to be implemented in order to make it easy for the companies to process the huge number of resumes that are received by the organizations.This technology converts an unstructured form of resume data into a structured data format. The resumes received are in the form of documents from which the data needs to be extracted first such that the text can be classified or predicted based on the requirements. A resume classification analyzes resume data and extracts the information into the machine readable output. It helps automatically store, organize, and analyze the resume data to find out the candidate for the particular job position and requirements. This thus helps the organizations eliminate the error-prone and time-consuming process of going through thousands of resumes manually and aids in improving the recruiters’ efficiency.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size:140%; '>  The basic data analysis process is performed such as data collection, data cleaning, exploratory data analysis, data visualization, and model building. The dataset consists of two columns, namely, Role Applied and Resume, where ‘role applied’ column is the domain field of the industry and ‘resume’ column consists of the text extracted from the resume document for each domain and industry. The aim of this project is achieved by performing the various data analytical methods and using the Machine Learning models and Natural Language Processing which will help in classifying the categories of the resume and building the Resume Classification Model.</p>", unsafe_allow_html=True)



#Exploratory data analaysis of resume 
if menu_id == 'Resume Data Analysis':
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1)


    st.subheader("🔍 Word Analysis")
    with st.expander('**Explore Data**'):
        x_axis = st.selectbox('**X-Axis**',options=[None]+list(word.columns),index=0)
        y_axis = st.selectbox('**Y-Axis**',options=[None]+list(word.columns),index=0)
        
        if x_axis and y_axis:
            if (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes != 'object'):
                plots = ['Scatter Plot']
            elif (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes == 'object'):
                plots = ['Bar Plot']
            elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes != 'object'):
                plots = ['Bar Plot']
            elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes == 'object'):
                plots = ['Bar Plot']

        elif x_axis and not y_axis:
            if word[x_axis].dtypes != 'object':
                plots = ['Histogram','Box Plot']
            else :
                plots = ['Bar Plot','Pie Plot']
        elif not x_axis and y_axis:
            if word[y_axis].dtypes != 'object':
                plots = ['Histogram','Box Plot']
            else :
                plots = ['Bar Plot','Pie Plot']
        else :
            plots = []
        if plots :
            disPlot = False
        else :
            disPlot = True

        disp = st.selectbox('**Plots**',options=plots,disabled=disPlot)
        if disp in ['Bar Plot','Pie Plot']:
            lim_dis = False
        else :
            lim_dis = True
        
        plot = st.button('**Plot**')
    
    if disPlot:
        st.warning('No Plots Available.')
    else :
        if plot :
            # plot here 
            if x_axis and not y_axis:
                if disp == 'Histogram':
                    fig = px.histogram(word,x=[x_axis],title=f'<b>{x_axis}</b>')
                    st.plotly_chart(fig)
                elif disp == 'Box Plot':
                    fig = px.box(word,x=[x_axis],title=f'<b>{x_axis}</b>')
                    st.plotly_chart(fig)
                elif disp == 'Bar Plot':
                    emp = word[x_axis].value_counts().head()
                    fig = px.bar(x=emp.index,y=emp.values)
                    st.plotly_chart(fig)
                elif disp == 'Pie Plot':
                    emp = word[x_axis].value_counts().head()
                    fig = px.pie(values=emp.values,names=emp.index,title=f'<b>{x_axis}</b>')
                    st.plotly_chart(fig)

            elif y_axis and not x_axis:
                if disp == 'Histogram':
                    fig = px.histogram(word,x=[y_axis],title=f'<b>{y_axis}</b>')
                    st.plotly_chart(fig)
                elif disp == 'Box Plot':
                    fig = px.box(word,x=[y_axis],title=f'<b>{y_axis}</b>')
                    st.plotly_chart(fig)
                elif disp == 'Bar Plot':    
                    emp = word[y_axis].value_counts().head()
                    fig = px.bar(x=emp.index,y=emp.values)
                    st.plotly_chart(fig)
                elif disp == 'Pie Plot':
                    emp = word[y_axis].value_counts().head()
                    fig = px.pie(values=emp.values,names=emp.index,title=f'<b>{y_axis}</b>')
                    st.plotly_chart(fig)    


            elif x_axis and y_axis:
                if (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes != 'object'):
                    if disp == 'Scatter Plot':
                        fig = px.scatter(word,x=x_axis,y=y_axis,title=f'{y_axis} Vs {x_axis}')
                        st.plotly_chart(fig)
                elif (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes == 'object'):
                    if disp == 'Bar Plot':
                        emp = word[[y_axis,x_axis]].groupby(by=[y_axis]).mean()
                        fig = px.bar(x=emp.values.ravel(),y=emp.index,title=f'{y_axis} Vs mean({x_axis})')
                        st.plotly_chart(fig)
                elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes != 'object'):
                    if disp == 'Bar Plot':
                        emp = word[[y_axis,x_axis]].groupby(by=[x_axis]).mean()
                        fig = px.bar(x=emp.index,y=emp.values.ravel(),title=f'{y_axis} Vs {x_axis}')
                        st.plotly_chart(fig)
                elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes == 'object'):
                    if disp == 'Bar Plot':
                        
                        #st.write(word[[x_axis,y_axis]].pivot_table(index=[x_axis,y_axis]).index)
                        word['dummy'] = np.ones(len(word))
                        emp = word[[x_axis,y_axis,'dummy']].pivot_table(index=[x_axis,y_axis],values=['dummy'],aggfunc=np.sum)
                        emp = emp.reset_index((0,1))

                        fig = px.bar(emp,x=x_axis,y='dummy',color=y_axis)
                        st.plotly_chart(fig)
                        
            else :
                st.warning('No Plots Available.')
        else :
            st.info('Click Plot Button')






### add skill screening and classification here
if menu_id == 'Resume Classification':
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1)


    # Function to remove punctuation and tokenize the text
    def tokenText(extText):
       
        # Remove punctuation marks
        punc = r'''!()-[]{};:'"\,.<>/?@#$%^&*_~'''
        for ele in extText:
            if ele in punc:
                puncText = extText.replace(ele, "")
                
        # Tokenize the text and remove stop words
        stop_words = set(stopwords.words('english'))
        puncText.split()
        word_tokens = word_tokenize(puncText)
        TokenizedText = [w for w in word_tokens if not w.lower() in stop_words]
        TokenizedText = []
      
        for w in word_tokens:
            if w not in stop_words:
                TokenizedText.append(w)
        return(TokenizedText)            

    # Function to extract Name and contact details
    def extract_name(Text):
        name = ''  
        for i in range(0,3):
            name = " ".join([name, Text[i]])
        return(name)

    def extract_skills(resume_text):

            nlp_text = nlp(resume_text)
            noun_chunks = nlp_text.noun_chunks

            # removing stop words and implementing word tokenization
            tokens = [token.text for token in nlp_text if not token.is_stop]
            
            # reading the csv file
            data = pd.read_csv('Resume raw_dataset.csv')   #for skills column
            
            # extract values
            skills = list(data.Skills.values)
            
            skillset = []
            
            # check for one-grams (example: python)
            for token in tokens:
                if token.lower() in skills:
                    skillset.append(token)
            
            # check for bi-grams and tri-grams (example: machine learning)
            for token in noun_chunks:
                token = token.text.lower().strip()
                if token in skills:
                    skillset.append(token)
            
            return [i.capitalize() for i in set([i.lower() for i in skillset])]

    def string_found(string1, string2):
            if re.search(r"\b" + re.escape(string1) + r"\b", string2):
                return True
            return False


    def extract_text_from_docx(path):
      if path.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            temp = docx2txt.process(path)
            return temp

    def display(docx_path):
        txt = docx2txt.process(docx_path)
        if txt:
            return txt.replace('\t', ' ')



    df = pd.DataFrame(columns=['Name','Skills'], dtype=object)

    col1,col2 = st.columns(2)

    with col1:
        st.title("Resume Classification")
            
        st.subheader('Upload Resume')
        upload_file = st.file_uploader('', type= ['docx'], accept_multiple_files=False)

        if upload_file is not None:
            displayed=display(upload_file)
            i=0
            text = extract_text_from_docx(upload_file)
            tokText = tokenText(text)
            df.loc[i,'Name']=extract_name(tokText)
            df.loc[i,'Skills']=extract_skills(text)
            st.header("**Resume Analysis**")
            st.success("Hello "+ df['Name'][0])
            try:        
                st.subheader('Name: '+ df['Name'][0])
            except:
                pass

            expander = st.expander("View Resume")
            expander.write(displayed)    

    with col2:
        st.header("**Skills Analysis**")
        keywords = st_tags(label='### Skills that'+ df['Name'][0] + ' have',
        text=' -- Skills', value=df['Skills'][0],key = '1')