# Chatbot COVID Generator

Generator of a COVID chatbot in English or Spanish for providing information about COVID.

## Project Report
Find the project report [here](./docs/fvilchez.pdf)

## Project Slides
Find the slides [here](https://docs.google.com/presentation/d/1evYVFMQyg-YX8XPEZF_AAR6oWqE7YpMreFF_qm9s6xM/edit?usp=sharing)

## Installation

Create your [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) environment based on the `spec-file.txt` file in the current repo with the following command:

`conda create --name <env> --file spec-file.txt`

Install packages from pip using the `requirements.txt` file

`pip install -r requirements.txt`

## Known Errors

For some reason, while running the Spanish chatbot, the translation API may fail in some occassions. Trying again a few more times will be enough to make it react again. You can comment line 37 & 38 in [./chatbot_generator/model/chatbot.py](./chatbot_generator/model/chatbot.py) if the API is down and is stopping testing.

## Chatbot Example for English - India

![](./docs/figures/India-EN-Chatbot.gif)

## Chatbot Example for Spanish - Peru

![](./docs/figures/Peru-ES-Chatbot.gif)
