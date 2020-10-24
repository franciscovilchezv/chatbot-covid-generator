import spacy
import requests
import chatbot_generator.model.parser as parser


nlp = spacy.load("en_core_web_sm")
REST_ACTION = "REST"
REDIRECTION = "REDIRECTION"

def extract_country(sentence):
    doc = nlp(sentence)

    for ent in doc.ents:
      if(ent.label_ == 'GPE'):
        return ent.text

    return sentence

def perform_action(action, question, personal_information):
  if (action['type'] == REST_ACTION):
    if(personal_information['country']):
      country = personal_information['country']
    else:
      country = extract_country(question)
  
    url = (action['endpoint'] % country)

    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data = payload)
    
    if response.status_code == 200:
      data = response.json()

      data_number = 0
      
      print(getattr(parser,action['parser'])(action['success_callback'], country, data, data_number))

      if("success_continuation" in action.keys()):
        while(data_number < len(data)):
          print(action['success_continuation'])

          if(input() == 'n'):
            data_number += 1
            print(getattr(parser,action['parser'])(action['success_callback'], country, data, data_number))
          else:
            break
      
    elif response.status_code == 404:
      print(action['not_found_callback'] % country)
    else:
      print(action['error_callback'])

    return 200
  elif (action['type'] == REDIRECTION):
    return 308
  else:
    raise Exception("Action %s not supported" % action["type"])