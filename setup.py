from chatbot_generator.model.chatbot import Chatbot

def main():
  cb = Chatbot()

  cb.train_model()
  print("The chatbot is ready :)")

  while(True):
    text = input()

    cb.ask(text)
    
if __name__ == "__main__":
  # execute only if run as a script
  main()