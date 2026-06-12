import os

def fix_typing_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '| None' in content or ' | ' in content:
        if 'from __future__ import annotations' not in content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('from __future__ import annotations\n' + content)
            print(f"Fixed {filepath}")

for root, dirs, files in os.walk('/Users/aksha/Desktop/Project/Elytics/backend'):
    for file in files:
        if file.endswith('.py'):
            fix_typing_in_file(os.path.join(root, file))
