import subprocess
import json
import os
import shutil

def test_cdif_expert():
    prompt = """Act as CDIF expert. Analyze the following dummy text:
From the Wikipedia page on Soil functions, here are all the extracted variables and measurements:
Variable Name 	Value 	Unit 	Context
Population in Soil Housing 	50 	Percentage 	Proportion of global population living in earth-based homes.
Key Soil Functions 	6 	Count 	Number of primary ecological and structural roles identified.
Bearing Strength 	Variable 	Pressure/Force 	Engineering capacity of soil to support structural loads.
Shear Strength 	Variable 	Force 	Resistance of soil to internal sliding failure (for roads/highways).
Carbon Sequestration Capacity 	Variable 	Mass/Volume 	Quantitative capacity of soil organic matter to store carbon.
CO2/O2 Exchange 	Variable 	Flux Rate 	Maintenance of air quantity/quality within the root zone.
Erosion Risk Index 	Variable 	Numerical Index 	Multi-property indicator derived from soil mapping and models.
Article Languages 	3 	Count 	Number of Wikipedia language versions for this topic.
Last Edited Date 	2025-10-30 	Date 	The most recent temporal update to the source material.
Pedotransfer Functions 	Model 	Mathematical 	Mathematical models used to infer complex soil properties.

These variables represent the functional, engineering, and ecological capacities of the soil system.
"""
    
    skills_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.agent/skills'))
    
    env = os.environ.copy()
    env['PATH'] = env.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin:/Users/vyacheslavtykhonov/.nvm/versions/node/v20.12.2/bin'

    gemini_path = shutil.which('gemini', path=env['PATH'])
    if not gemini_path:
        for p in ['/opt/homebrew/bin/gemini', '/usr/local/bin/gemini', '/usr/bin/gemini', os.path.expanduser('~/.local/bin/gemini')]:
            if os.path.exists(p):
                gemini_path = p
                break
        else:
            gemini_path = 'gemini'
            
    cmd = [
        gemini_path,
        '-p', prompt,
        '--yolo',
        '-o', 'json',
        '--policy', skills_dir
    ]
    
    print("Running Gemini CLI... (this may take a moment)")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    try:
        # Extract the JSON response
        data = json.loads(result.stdout)
        print("\n--- LLM Final Response ---")
        print(data.get("response", "No response found in JSON."))
    except json.JSONDecodeError:
        print("\n--- STDOUT (Failed to parse JSON) ---")
        print(result.stdout)
        print("\n--- STDERR ---")
        print(result.stderr)
    
if __name__ == '__main__':
    test_cdif_expert()
