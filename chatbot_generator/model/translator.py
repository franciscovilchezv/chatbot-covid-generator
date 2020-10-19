import json

# https://www.tutorialspoint.com/python_text_processing/python_text_translation.htm
# This tool has a limit quota
# from translate import Translator
# translator = Translator(to_lang=languages[lang])

# https://pypi.org/project/googletrans/
# Used as an alternative since the other one has a limit quota
from googletrans import Translator
translator = Translator()

languages = {
  "ES": "Spanish"
}

def translate_sentence(sentence, dest_language):
  return translator.translate(sentence, dest_language).text.replace("% s", " %s")

def generate_dataset(lang="ES"):
  print("Generating dataset...")
  
  dataset = 'chatbot_generator/dataset/covid_intents_EN.json'

  with open(dataset) as json_data:
    data = json.load(json_data)
    intents = data['intents']
    total_intents = len(intents)
    for intent in intents:
      print("Translating intents (%s left)" % (total_intents))
      total_intents = total_intents - 1

      patterns_translated = []
      for pattern in intent['patterns']:
        patterns_translated.append(translate_sentence(pattern, lang))
      intent['patterns'] = patterns_translated

      responses_translated = []
      for response in intent['responses']:
        responses_translated.append(translate_sentence(response, lang))
      intent['responses'] = responses_translated

      if intent['action']:
        intent['action']['success_callback'] = translate_sentence(intent['action']['success_callback'], lang)
        intent['action']['not_found_callback'] = translate_sentence(intent['action']['not_found_callback'], lang)
        intent['action']['error_callback'] = translate_sentence(intent['action']['error_callback'], lang)

    print("Saving intents file...")
    with open("chatbot_generator/dataset/covid_intents_ES.json", "w", encoding='utf8') as outfile:  
      json.dump(data, outfile, ensure_ascii=False, indent=2) 

  print("Dataset created!")