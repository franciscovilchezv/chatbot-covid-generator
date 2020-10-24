from chatbot_generator.model.chatbot import Chatbot
import os

welcome_messages = {
  "EN": "Welcome to the COVID chatbot",
  "ES": "Bienvenido al COVID chatbot"
}

redirection_message = {
  "EN": "Returning to Chabot Generator Configuration Mode",
  "ES": "Regresando al modo de configuración de Generación de Chatbot"
}

def main():
  
  while(True):
    print("Welcome to COVID Chatbot Generator. Create your chatbot as desired.")
    print("Type `quit` to exit or press `enter` to continue")

    if(input() == "quit"):
      break
    
    # (lang, country) = ("EN", "India")
    (lang, country) = get_input_parameters()
    cb = Chatbot(lang, country)

    cb.train_model()
    os.system("clear")

    print(welcome_messages[lang], end="\n\n")

    while(True):
      print("User\t\t\t$: ", end='')
      text = input()

      response_code = cb.ask(text)

      if response_code == 308:
        print(redirection_message[lang], end="\n\n")
        break

  print("Exiting COVID Chatbot Generator...")

def get_input_parameters():
  print("System (Covid:Config)\t$: Type your language preference (EN/ES)")
  print("User\t\t\t$: ", end='')
  lang = input()
  print()

  print("System (Covid:Config)\t$: Type your country:")
  print("User\t\t\t$: ", end='')
  country = input()
  print()

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