"""
Spark NLP NER model for Hebrew 
https://nlp.johnsnowlabs.com/2020/12/09/hebrewner_cc_300d_he.html
"""

#import sparknlp


# word_embeddings = WordEmbeddingsModel.pretrained("hebrew_cc_300d", "he") \
# .setInputCols(["document", "token"]) \
# .setOutputCol("embeddings")
# ner = NerDLModel.pretrained("hebrewner_cc_300d", "he") \
# .setInputCols(["sentence", "token", "embeddings"]) \
# .setOutputCol("ner")
# ner_converter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")
# nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter])
# light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
# annotations = light_pipeline.fullAnnotate("""
# ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , 
# שגויס לארגון הפת"ח והופעל על ידי חיזבאללה. 
# אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל , לבצע פיגוע ברכבת ישראל בנהריה , 
# לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים
# """


import nlu
result =nlu.load("he.ner").predict("""ח והופעל על ידי חיזבאללה. 
אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל
 , לבצע פיגוע ברכבת ישראל בנהריה ,
 לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים.""")

print("End")

 