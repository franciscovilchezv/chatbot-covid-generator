import spacy
import requests
import chatbot_generator.model.parser as parser


nlp = spacy.load("en_core_web_sm")
REST_ACTION = "REST"

def extract_country(sentence):
    doc = nlp(sentence)

    for ent in doc.ents:
      if(ent.label_ == 'GPE'):
        return ent.text

    return sentence

def perform_action(action, question, personal_information):
  if(personal_information['country']):
    country = personal_information['country']
  else:
    country = extract_country(question)
  if (action['type'] == REST_ACTION):
    url = (action['endpoint'] % country)

    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data = payload)
    
    if response.status_code == 200:
      data = response.json()
      
      print(getattr(parser,action['parser'])(action['success_callback'], country, data))
    elif response.status_code == 404:
      print(action['not_found_callback'] % country)
    else:
      print(action['error_callback'])