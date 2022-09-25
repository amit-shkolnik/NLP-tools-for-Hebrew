"""
https://pypi.org/project/hebspacy/0.1.7/

"""
import spacy
import sklearn
print(f"sklearn: {sklearn.__version__}")
import pandas
print(f"pandas: {pandas.__version__}")
nlp = spacy.load("he_ner_news_trf")
text = """מרגלית דהן
מספר זהות 11278904-5

2/12/2001
ביקור חוזר מ18.11.2001
במסגרת בירור פלפיטציות ואי סבילות למאמצים,מנורגיות קשות ע"ר שרירנים- ביצעה מעבדה שהדגימה:
המוגלובין 9, מיקרוציטי היפוכרומטי עם RDW 19,
פריטין 10, סטורציית טרנספרין 8%. 
מבחינת עומס נגיפי HIV- undetectable ומקפידה על HAART
"""

text="""
עוד אסונות טבע, בצל האזהרות מהשלכות שינויי האקלים:
 באלג'יריה נספו לפחות 38 איש בשריפות יער ענקיות בצפון המדינה. "זה היה טורנדו של אש",
 דיווח עיתונאי מקומי. באי הצרפתי קורסיקה נספו שלושה, בהם בת 13, בסופה פתאומית עם רוחות של עד 220 קמ"ש,
 ובצפון איטליה ברד ענקי הרס יבולים שלמים. בסין דיווחו על 16 הרוגים בשיטפונות ענק - לצד בצורת היסטורית
"""

doc = nlp(text)
for entity in doc.ents:
    print(f"{entity.text} \t {entity.label_}: {entity._.confidence_score:.4f} ({entity.start_char},{entity.end_char})")

print("END")