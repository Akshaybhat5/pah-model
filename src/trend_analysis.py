import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st
import zipfile
import docx
import fitz  # PyMuPDF for PDF processing
import io
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import plotly.express as px
import cloudpickle
import plotly.graph_objects as go
import time
import altair as alt
import requests
import json



#function to read pdf
def read_pdf(file):
    content = ""
    with fitz.open('pdf', file.read()) as doc:
        for page in doc:
            content += page.get_text()
        content = pd.DataFrame([content], columns=['page_content'])
    return content

#functin to read docx
def read_docx(file):
    doc = docx.Document(file)
    content = "\n".join([parra.text for parra in doc.paragraphs])
    content = pd.DataFrame([content], columns=['page_content'])
    return content

#animation
def display_result_animation():
    placeholder = st.empty()

    # Simulating a loading animation
    animation = ["Analyzing.", "Analyzing..", "Analyzing...", "Analyzing....", "Analyzing....."]
    for i in range(5):
        placeholder.markdown(f"<h3 style='font-size:24px; color: black;'>{animation[i % len(animation)]}</h3>", unsafe_allow_html=True)

        time.sleep(0.5)
    placeholder.empty()

#main trend analysis function
def trend_analysis():

    """
        Explain this function ----------------------> later
    """
    url = 'https://pah-model-hlutv.australiasoutheast.inference.ml.azure.com/score'
    api_key = "5NONOeVzuAbouO1p3Wix0XUFusB2Mes8"  # Replace with your Azure ML service key
    headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
    uploaded_zip = st.file_uploader("Upload a zipped folder", type=["zip"])

    document_names = []
    outcomes = []
    outcomes_probability = []
    if uploaded_zip:
        st.success('Zip uploaded successfully.')
        display_result_animation()
        display_result_animation()
        display_result_animation()
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            file_list = zip_ref.namelist()
                
            for file_name in file_list[1:]:
                document_names.append(file_name)
                with zip_ref.open(file_name) as file:
                    # st.write(f"Processing file: {file_name}")
                        
                        # Process each file based on extension
                    if file_name.endswith(".pdf"):
                        content = read_pdf(file)
                        response = requests.post(url, data=content.to_json(orient="records"), headers=headers)
            
                        if response.status_code == 200:
                            response = response.json()
                            response_dict = json.loads(response)
                            
                            display_result_animation()
                        # outcomes.append(random.choice(choices_outcome))
                    elif file_name.endswith(".docx"):
                        content = read_docx(file)
                        response = requests.post(url, data=content.to_json(orient="records"), headers=headers)
            
                        if response.status_code == 200:
                            response = response.json()
                            response_dict = json.loads(response)
                            display_result_animation()
                    else:
                        st.write(f"Unsupported file format: {file_name}")
                        continue


                    model_predictions = response_dict.get('predictions')[0]
                    model_prediction_proba = response_dict.get('probabilities')[0]
                    if model_predictions == "Pedagogy":
                        outcomes.append("Dependent Learner")
                    elif model_predictions == "Andragogy":
                        outcomes.append("Independent Learner")
                    elif model_predictions == "Heutagogy":
                        outcomes.append("Self-Determined Learner")

                    outcomes_probability.append(model_prediction_proba)
                    

    doc_name_lst = []
    for names in document_names:
        doc_name = names.split('/')[1]
        doc_name_lst.append(doc_name)

    def PAH_conversion(lst:list):
        pah_conversion = []
        for outcome in lst:
            if outcome == 'Dependent Learner':
                pah_conversion.append(1)
            elif outcome == 'Independent Learner':
                pah_conversion.append(2)
            elif outcome == 'Self-Determined Learner':
                pah_conversion.append(3)
        return pah_conversion
            

    values_conversion = PAH_conversion(outcomes)

    metadata = []
    for names, outcome, outcome_name in zip(doc_name_lst, values_conversion, outcomes):
        metadata.append({
            'Documents': names.split('.')[-2:][0],
            'Outcome Value':outcome,
            'Outcome Category':outcome_name
        })
    metadata_df = pd.DataFrame(metadata)
    # metadata_df = metadata_df.sort_values(by='Documents')
    if len(metadata_df) >0:
        metadata_df['Document_Number'] = metadata_df['Documents'].str.extract(r'(\d+)').astype(int)
        metadata_df.sort_values(by='Document_Number', inplace=True)

# Drop the temporary sorting column if you don’t need it
        metadata_df.drop(columns='Document_Number', inplace=True)

# Reset index if needed
        metadata_df.reset_index(drop=True, inplace=True)

    probability = pd.DataFrame(outcomes_probability, columns=['Independent Learner', 'Self-Determined Learner', 'Dependent Learner'])
    file_name = pd.DataFrame(doc_name_lst, columns=['doc_name'])

    df = pd.concat([file_name, probability], axis=1)
    df = df.melt(id_vars='doc_name', var_name='category', value_name='value')
    df = df.groupby(['doc_name', 'category'])['value'].apply(list).reset_index()
    df['value'] = df['value'].apply(lambda x: round(x[0]*100, 2))

    if len(outcomes)>0:
        color_map = {'Independent Learner': 'orange', 'Self-Determined Learner': 'green', 'Dependent Learner': 'red'}

# Create a continuous line connecting all points (ignoring the categories)
        fig = go.Figure()

# Add a single continuous line connecting all points
        fig.add_trace(go.Scatter(
            x=metadata_df['Documents'], 
            y=metadata_df['Outcome Value'], 
            mode='lines',  # Just lines connecting all points
            line=dict(color='black'),  # Line color (gray to avoid conflict with categories)
            name='Line Connecting All Points',
            showlegend=False,
            hoverinfo='none'
            ))

#  Overlay the colored markers based on the category (outcome_name)
        for outcome, color in color_map.items():
    # Filter points based on the outcome_name and color them
            df_filtered = metadata_df[metadata_df['Outcome Category'] == outcome]
            fig.add_trace(go.Scatter(
                x=df_filtered['Documents'],
                y=df_filtered['Outcome Value'],
                mode='markers',  # Only markers for the categories
                name=outcome,
                marker=dict(color=color, size=10),  # Marker size and color based on outcome_name
                showlegend=True,
                hoverinfo='none'
            ))

# Step 3: Add labels for each point (annotations)
        for i, row in metadata_df.iterrows():
            fig.add_annotation(
            x=row['Documents'],
            y=row['Outcome Value'],
            text=row['Outcome Category'],
            showarrow=True,
            arrowhead=3,
            ax=20,
            ay=-30,
            font=dict(color='blue'),  # Change annotation text color (labels)
            arrowcolor='black'  # Change arrow color
            )

# Step 4: Customize layout (removing gridlines and axis lines)
        fig.update_layout(
            title={
                'text': "Document Trend Analysis",
                'font': {'size': 22, 'color': 'black'},  # Title font color and size
            },
        xaxis_title={
            'text': "Document Name",
            'font': {'size': 18, 'color': 'black'},  # X-axis title font color and size
        },
        yaxis_title={
            'text': "Outcome",
            'font': {'size': 18, 'color': 'black'},  # Y-axis title font color and size
        },
        legend_title={
            'text': "Document Outcome",
            'font': {'size': 14, 'color': 'black'},  # Legend title font color
        },
    
    # Customizing the axis tick labels color
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showline=False, 
            tickfont=dict(color='black'),
            categoryorder='array',
            categoryarray=metadata_df['Documents'].tolist()
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showline=False, 
            tickvals=[0, 1, 2, 3],
            tickfont=dict(color='black')  # Y-axis tick labels color
        ),
    
    # Customize the legend text color
        legend=dict(
            font=dict(color='black')  # Set the legend text color
        ),

        plot_bgcolor='#FFFFFF',  # Background behind the plot
        paper_bgcolor='#FFFFFF',  # Background behind the whole chart
    )

# Show the plot
    # fig.show()
        migrations_col = st.columns((0.5, 0.5))
        with migrations_col[0]:
            st.plotly_chart(fig, config={'displayModeBar': False})
        with migrations_col[1]:
            color_scale = alt.Scale(domain=['Independent Learner', 'Self-Determined Learner', 'Dependent Learner'],
                        range=["orange", "green", "red"])  # Specify colors for each category

# Altair Bar Chart
            chart = alt.Chart(df).mark_bar().encode(
                                    y=alt.X('value:Q', title='Confidence Score',axis=alt.Axis(grid=False)),
                                    x=alt.Y('doc_name:N', title='Document Name',axis=alt.Axis(grid=False)),
                                    color=alt.Color('category:N', title='Category', scale=color_scale),
                                            tooltip=[
                                                        alt.Tooltip('doc_name:N', title='Document Name'),
                                                        alt.Tooltip('category:N', title='Category'),
                                                        alt.Tooltip('value:Q', title='Confidence Score in (%)')
                                                                                    ]
                                                        ).properties(
                                                            title=alt.TitleParams(
                                                                text="Document Confidence Breakdown",
                                                                fontSize=20,
                                                                color='white',
                                                                subtitle="Outcome by Document"
                                                            ),
                                                            width=600,
                                                            height=400,
                                                            background='#08182E',  # Chart background color
                                                            padding={'top': 20, 'bottom': 20, 'left': 20, 'right': 20}
                                                            ).configure_axis(
                                                                labelColor='white',
                                                                titleColor='white',
                                                                grid=False,
                                                                domain=False
                                                            ).configure_legend(
                                                                titleColor='white',
                                                                labelColor='white'
                                                            ).configure_title(
                                                                fontSize=24,
                                                                font='Arial',
                                                                color='white'
                                                            )
            st.altair_chart(chart, use_container_width=True)



hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

