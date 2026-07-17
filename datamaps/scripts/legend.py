import xml.etree.ElementTree as ET
import json

svg_file = "map_clean.svg"
tree = ET.parse(svg_file)
root = tree.getroot()

# Namespaces can be annoying; strip them for easier XPath
ns = {"svg": "http://www.w3.org/2000/svg"}
for elem in root.iter():
    if '}' in elem.tag:
        elem.tag = elem.tag.split('}', 1)[1]

legend_items = []

# Find all groups that contain a rectangle (colour) and text (value)
for g in root.findall(".//g"):
    rect = g.find("rect")
    txt = g.find("text")
    if rect is not None and txt is not None:
        color = rect.get('fill')
        value_text = "".join(txt.itertext()).strip()
        legend_items.append({"color": color, "value": value_text})

# Pretty‑print as JSON
legend_json = json.dumps(legend_items, indent=2)
print("Legend (colour → value):")
print(legend_json)

# Save to file if you want
with open("legend.json", "w") as f:
    f.write(legend_json)
