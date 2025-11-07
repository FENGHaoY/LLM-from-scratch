import re
with open('the-verdict.txt','r', encoding='utf-8') as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.!?:_"()\']|--|\s)', raw_text)
preprocessed = [token for token in preprocessed if token.strip()]

print(preprocessed[:20])