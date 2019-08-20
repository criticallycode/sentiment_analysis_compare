import custom_sentiment_analyzer as s
import os

os.chdir('C:/PycharmProjects/Projects/sentiment_analysis2/')

text = "This game is terrible. The controls are garbage and so are the character models, I hate everything about it."
text2 = "I love her so much. She makes me really happy."

analyzer = s.sentiment(text)
analyzer2 = s.sentiment(text2)

print(analyzer)
print(analyzer2)