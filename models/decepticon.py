from transformers import pipeline

tweet = 'RT @AJA_Cortes: $BTC dropped to $3k last year and everyone was fawking crying and having meltdowns'

nlp = pipeline(task='text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')

print(f'Result: {nlp(tweet)}')