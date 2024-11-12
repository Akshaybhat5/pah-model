import streamlit as st
from streamlit_option_menu import option_menu
import docx
import fitz
from streamlit_lottie import st_lottie
import time
import json
from streamlit.components.v1 import html
from ipyvizzu import Chart, Data, Config, Style, DisplayTarget
import pandas as pd
import random
import zipfile
import plotly.express as px
 # use this for single document analysis
import mlflow
import sklearn
import cloudpickle
import altair as alt
import requests



#read pdf file
def read_pdf(file):
    content = ""
    with fitz.open('pdf', file.read()) as doc:
        for page in doc:
            content += page.get_text()
        content = pd.DataFrame([content], columns=['page_content'])
    return content


#read docx file
def read_docx(file):
    doc = docx.Document(file)
    content = "\n".join([parra.text for parra in doc.paragraphs])
    content = pd.DataFrame([content], columns=['page_content'])
    return content


#animation for document analyzing 
def display_result_animation():
    placeholder = st.empty()

    # Simulating a loading animation
    animation = ["Analyzing.", "Analyzing..", "Analyzing...", "Analyzing....", "Analyzing....."]
    for i in range(5):  # Loop for animation
        placeholder.markdown(f"<h3 style='font-size:24px; color: black;'>{animation[i % len(animation)]}</h3>", unsafe_allow_html=True)
        time.sleep(0.5)
    placeholder.empty()



alt.renderers.set_embed_options(actions=False)

#altair donut chart for confidence score breakdown
def make_donut(input_response, input_text, input_color):
    """
            Explain this function --------------------> Later    
    """
    
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    
    source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
    source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
    plot = alt.Chart(source).mark_arc(innerRadius=60, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=200, height=180)
    
    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response:.2f} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=60, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=200, height=180)
    

    return plot_bg + plot + text


 
# main document analysis function
def document_analysis():
    """
    Document analysis function for Streamlit
    """
    url = 'https://pah-model-hlutv.australiasoutheast.inference.ml.azure.com/score'
    api_key = "5NONOeVzuAbouO1p3Wix0XUFusB2Mes8" # Replace with your Azure ML service key
    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
    
    information = st.file_uploader('Please upload files in pdf or document format', type=['pdf', 'docx'])
    if information:
        st.success('File uploaded successfully.')
        
        # Debug step: show file name
        st.write("Uploaded file name:", information.name)
        
        # PDF Handling
        if information.name.endswith('.pdf'):
            content = read_pdf(information)

            response = requests.post(url, data=content.to_json(orient="records"), headers=headers)
            
            if response.status_code == 200:
                response = response.json()
                response_dict = json.loads(response)
                model_predictions = response_dict.get('predictions')[0]

            display_result_animation()
            
            if model_predictions == "Pedagogy":
                model_predictions = 'Dependent Learner'
                final_outcome_text = f"""
                                    <div style="font-size:25px;">
                                        AI-Generated Label:  <span style="color:red; font-weight:bold;">{model_predictions}</span>
                                    </div>
                                """
            elif model_predictions == "Heutagogy":
                model_predictions = 'Self-determined Learner'
                final_outcome_text = f"""
                                    <div style="font-size:25px;">
                                        AI-Generated Label:  <span style="color:green; font-weight:bold;">{model_predictions}</span>
                                    </div>
                                """
            elif model_predictions == "Andragogy":
                model_predictions = 'Independent Learner'
                final_outcome_text = f"""
                                    <div style="font-size:25px;">
                                        AI-Generated Label:  <span style="color:orange; font-weight:bold;">{model_predictions}</span>
                                    </div>
                                """
                

    # # Display the styled prediction
            st.markdown(final_outcome_text, unsafe_allow_html=True)

            Independent_learner, Self_determined_Learner, Dependent_learner = [i for i in response_dict.get('probabilities')[0]]

            Independent_learner = round(Independent_learner*100, 2)
            Self_determined_Learner = round(Self_determined_Learner*100, 2)
            Dependent_learner = round(Dependent_learner*100, 2)

            st.markdown('---')
            st.markdown('##### Confidence Score Breakdown')
            migrations_col = st.columns((0.3, 0.3, 0.4))
            with migrations_col[0]:
        
                st.write('Independent Learner')
                chart = make_donut(Independent_learner, 'Independent Learner', 'orange')
                st.altair_chart(chart)
        # migrations_col[0].metric(st.altair_chart(chart), 75)
            with migrations_col[1]:
                st.write('Self-determined Learner')
                chart_2 = make_donut(Self_determined_Learner, 'Self-determined Learner', 'green')
                st.altair_chart(chart_2)
            with migrations_col[2]:
                st.write('Dependent Learner')
                chart_3 = make_donut(Dependent_learner, 'Dependent Learner', 'red')
                st.altair_chart(chart_3)


            #DOCX
        elif information.name.endswith('.docx'):
            content = read_docx(information)
            response = requests.post(url, data=content.to_json(orient="records"), headers=headers)
            
            if response.status_code == 200:
                response = response.json()
                response_dict = json.loads(response)
                model_predictions = response_dict.get('predictions')[0]
          
    
            display_result_animation()

            if model_predictions == "Pedagogy":
                model_predictions = 'Dependent Learner'
                final_outcome_text = f"""
                                    <div style="font-size:25px;">
                                        AI-Generated Label:  <span style="color:red; font-weight:bold;">{model_predictions}</span>
                                    </div>
                                """
            elif model_predictions == "Heutagogy":
                model_predictions = 'Self-determined Learner'
                final_outcome_text = f"""
                                    <div style="font-size:25px;">
                                        AI-Generated Label:  <span style="color:green; font-weight:bold;">{model_predictions}</span>
                                    </div>
                                """
            elif model_predictions == "Andragogy":
                model_predictions = 'Independent Learner'
                final_outcome_text = f"""
                                    <div style="font-size:25px;">
                                        AI-Generated Label:  <span style="color:orange; font-weight:bold;">{model_predictions}</span>
                                    </div>
                                """
    
    # Display the styled prediction
            st.markdown(final_outcome_text, unsafe_allow_html=True)

            Independent_learner, Self_determined_Learner, Dependent_learner = [i for i in response_dict.get('probabilities')[0]]

            Independent_learner = round(Independent_learner*100, 2)
            Self_determined_Learner = round(Self_determined_Learner*100, 2)
            Dependent_learner = round(Dependent_learner*100, 2)

            st.markdown('---')
            st.markdown('##### Confidence Score Breakdown')
            migrations_col = st.columns((0.3, 0.3, 0.4))
            with migrations_col[0]:
        
                st.write('Independent Learner')
                chart = make_donut(Independent_learner, 'Independent Learner', 'orange')
                st.altair_chart(chart)
        # migrations_col[0].metric(st.altair_chart(chart), 75)
            with migrations_col[1]:
                st.write('Self-determined Learner')
                chart_2 = make_donut(Self_determined_Learner, 'Self-determined Learner', 'green')
                st.altair_chart(chart_2)
            with migrations_col[2]:
                st.write('Dependent Learner')
                chart_3 = make_donut(Dependent_learner, 'Dependent Learner', 'red')
                st.altair_chart(chart_3)
                
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)
