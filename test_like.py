import sqlite3

conn = sqlite3.connect('storage/materials.db')
cursor = conn.cursor()

# 测试不同的LIKE模式
patterns = [
    '%权%',
    '% 权 %',
    '%权 益%',
    '% 权 益 %',
]

for pattern in patterns:
    cursor.execute("SELECT COUNT(*) FROM assets WHERE asr_text LIKE ?", (pattern,))
    count = cursor.fetchone()[0]
    print(f"Pattern '{pattern}': {count} results")

conn.close()
