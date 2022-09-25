
"""
Demo of using STANZA - https://stanfordnlp.github.io/stanza/
Stanza A Python NLP Package for Many Human Languages

Stanza is a collection of accurate and efficient tools for the linguistic analysis 
of many human languages. Starting from raw text to syntactic analysis and entity recognition, 
Stanza brings state-of-the-art NLP models to languages of your choosing.
"""

import stanza
print("START")
stanza.download('he') # download English model
nlp = stanza.Pipeline('en') # initialize English neural pipeline
doc = nlp("ראש הממשלה בנימין נתניהו תוקף את איראן על יוזמת הגרעין שפורסמה בוינה") 
print(doc)
print("END")
