import requests

url = "https://cdn-lfs.hf.co/repos/42/7f/427f7497b6c6596c18b46d5a72e61364fcad12aa433c60a0dbd4d344477b9d81/c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27TinyStories-train.txt%3B+filename%3D%22TinyStories-train.txt%22%3B&response-content-type=text%2Fplain&Expires=1743953322&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0Mzk1MzMyMn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy80Mi83Zi80MjdmNzQ5N2I2YzY1OTZjMThiNDZkNWE3MmU2MTM2NGZjYWQxMmFhNDMzYzYwYTBkYmQ0ZDM0NDQ3N2I5ZDgxL2M1Y2Y1ZTIyZmYxMzYxNGU4MzBhZmJlNjFhOTlmYmNiZThiY2I3ZGQ3MjI1MmI5ODlmYTExMTdhMzY4ZDQwMWY%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=uwqntxvD6rYeDFYvTJUSd1%7E0WOmGxOHrAatzoyvpBiLnL9Zkzr2VILMKcNlWZSLsssRSaJ58GPCsmbaOnaPlD-WPOgI%7EKfkJnlFe89zQ4GDnakolMjBCzeeMjPOB6lorpefzcvrx0VxuBzUdFHuEhmQ2n5BTJgx5bZyPu3494085ezd25Lu22O2Leakz7rzPQdVsPu392WrXSoozIvzHXxODFe7fH2GqZ6LSXe48t1CmjjNrxhjbIgCVZVa4f7lNOrvuIJGEfgKQG0BB8crymxiOGMbzFH8ZqnHuiX98Iel9LHCUrj8bOEYqX9SYxqWjmDoy7wjS2RU8e8YpgttmDw__&Key-Pair-Id=K3RPWS32NSSJCE"

response = requests.get(url)
text = response.text
num_chunks = len(text) // 1000
with open('TinyStories-train.txt', 'w', encoding='utf-8') as f:
    for i in range(num_chunks):
        chunk = text[i * 1000:(i + 1) * 1000]
        f.write(chunk + '\n')
    if len(text) % 1000 != 0:
        f.write(text[num_chunks * 1000:] + '\n')

print("Dataset downloaded and saved as 'TinyStories-train.txt'")