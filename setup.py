from chatbot_generator.model.chatbot import Chatbot


def main():
  # (lang, country) = get_input_parameters()
  # cb = Chatbot(lang, country)

  # Test mode
  cb = Chatbot()

  cb.train_model()

  while(True):
    text = input()

    cb.ask(text)

def get_input_parameters():
  print("Type your langage preference (EN/ES):")
  lang = input()

  print("Type your country:")
  country = input()

  return lang, country
    
if __name__ == "__main__":
  # execute only if run as a script
  main()

"""
config: 
      a) Make categories of questions:chitchat, covid cases, health services, prevention
      b) Decided points of customization - language, location
      c) Write program to generate intent file based on customization points
      d) Automate process to create chatbot
"""