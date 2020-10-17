import pandas as pd
import spacy
import requests


nlp = spacy.load("en_core_web_sm")
REST_ACTION = "REST"

def extract_country(sentence):
    doc = nlp(sentence)

    for ent in doc.ents:
      if(ent.label_ == 'GPE'):
        return ent.text

    return sentence

def perform_action(action, question):
  country = extract_country(question)
  if (action['type'] == REST_ACTION):
    url = (action['endpoint'] % country)

    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data = payload)
    
    if response.status_code == 200:
      data = response.json()
      
      print(action['success_callback'] %
        (
          data['country'],
          pd.to_datetime(data['updated'], unit='ms').date(),
          data['todayCases'],
          data['cases']
        )
      )
    elif response.status_code == 404:
      print(action['not_found_callback'] % country)
    else:
      print(action['error_callback'])