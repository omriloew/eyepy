import xml.etree.ElementTree as ET
from pathlib import Path

mff_dir = Path(r"C:\Users\may\Desktop\pain_test\eyepy\log_files\omer\10\plr\PLR_1_20260115_113336.mff")
xml_path = mff_dir / "pnsSet.xml"

tree = ET.parse(xml_path)
root = tree.getroot()

# Print all short text nodes that look like channel labels
texts = []
for elem in root.iter():
    if elem.text:
        t = elem.text.strip()
        if 1 <= len(t) <= 40:
            texts.append((elem.tag, t))

# show unique values preserving order
seen = set()
uniq = []
for tag, t in texts:
    if t not in seen:
        seen.add(t)
        uniq.append((tag, t))

print("First 200 unique text nodes from pnsSet.xml:")
for tag, t in uniq[:200]:
    print(f"{tag}: {t}")
