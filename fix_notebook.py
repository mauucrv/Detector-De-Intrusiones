import json

path = r'notebooks/01_exploratory_data_analysis.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if ".cat.remove_unused_categories()" in line and "astype" not in line:
                cell['source'][i] = line.replace(
                    "['Label'].cat.remove_unused_categories()",
                    "['Label'].astype('category').cat.remove_unused_categories()"
                )
                count += 1

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f'Fixed {count} occurrences.')
