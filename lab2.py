import nltk
nltk.download("all")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

text = "В процессе работы (чистка/мойка поверхности) работает тихо и не заметно, однако в момент завершения работы когда производится чистка пылесоса (забор мусора) шумит сильно, но не долго.Стоит своих денег. Полностью оправдал мои ожидания."

sentiment_pipeline = pipeline("sentiment-analysis")
print(sentiment_pipeline(text))


words = word_tokenize(text)
stop_words = stopwords.words("russian")
filtered_words = list(filter(lambda word: word not in stop_words, words))

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(word))

stemmer = SnowballStemmer(language="russian")
stemmed_words = []
for word in lemmatized_words:
    stemmed_words.append(stemmer.stem(word))

print("-------------------------------------------")

sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline(stemmed_words)
negative_count = 0
positive_count = 0

for word in result:
    if word['label'] == 'POSITIVE':
        positive_count += 1
    else:
        negative_count += 1

print(f"{negative_count} - Negative")
print(f"{positive_count} - Positive")

print("-------------------NLTK-----------------")


nltk_analyzer = SentimentIntensityAnalyzer()
print(nltk_analyzer.polarity_scores(text))

import spacy
import textacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()

article = """London is the capital and most populous city of England and  the United Kingdom.
Standing on the River Thames in the south east of the island of Great Britain,
London has been a major settlement  for two millennia.  It was founded by the Romans,
who named it Londinium."""

article_with_mistakes = """London is the capidftal and most, popuus city, of Engdfland and  the United Kingdom.
Standing on the River Thadmes in the soudth east of thdfe island of Great Britadfin,
London has been a majdor, settlement,  for two mildlennia.  It was founded by the Romans,
who named it Londiniudm."""




def textacy_nlp(text):
    doc = nlp(text)
    for entity in doc.ents:
        print(f"{entity.text} ({entity.label_})")
    statements = textacy.extract.semistructured_statements(doclike = doc, entity = "London", cue = "be")
    print(statements)
    for statement in statements:
        subject, verb, fact = statement
        print(f"- {fact}")


print('-------------------article-------------------')
textacy_nlp(article)

print('-------------------article with mistakes----------')
textacy_nlp(article_with_mistakes)
