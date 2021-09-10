import streamlit as st
import streamlit.components.v1 as components

from sparknlp_display import RelationExtractionVisualizer
from sparknlp_display.converter import pkre2sparknlp 

import json

with open('pkre_sample.json', 'r') as f:
	pkre_sample = json.load(f)

def predict(text):
	"""
	Relation extraction
	"""
	# TODO: insert predicting script here
	# simulate prediction
	pkre_pred = pkre_sample
	
	# convert to sparknlp format
	pkre = pkre2sparknlp(pkre_pred)

	return pkre


def display(text):
	pkre = predict(text)

	visualizer = RelationExtractionVisualizer()
	html_content = visualizer.display(pkre,
					relation_col='relations',
					document_col='document',
					show_relations=True,
					return_html=True)

	html_content = '<div>' + html_content + '</div>'
	components.html(html_content, width=1100, height=500, scrolling=True)


## MAIN APP
st.title('PK Entity Relation Extraction App')

text = st.text_area('Input text', pkre_sample['text'], height=150)

is_predict = st.button('Predict')

if is_predict:
	display(text)
