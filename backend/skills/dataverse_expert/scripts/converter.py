import json

def croissant_to_dataverse(croissant_data, config=None):
    """
    Converts a Croissant JSON object to a Dataverse-compliant JSON-LD dictionary.
    """
    if config is None:
        config = {
            "authorName": "The Minority Report",
            "authorAffiliation": "United Nations Office for Disaster Risk Reduction",
            "datasetContactEmail": "dummy@email.com",
            "datasetContactName": "The Minority Report Team",
            "subject": "Earth and Environmental Sciences",
            "license_url": "http://creativecommons.org/licenses/by/4.0",
            "license_name": "CC BY 4.0",
            "license_rights": "Creative Commons Attribution 4.0 International"
        }

    voc = {}

    # Title
    voc['http://purl.org/dc/terms/title'] = croissant_data.get('name', 'Untitled Dataset')
    
    # Restrictions (Default)
    voc["https://dataverse.org/schema/core#restrictions"] = "No restrictions"
    
    # Subject
    voc['http://purl.org/dc/terms/subject'] = config.get('subject', "Earth and Environmental Sciences")

    # Creator
    creator = {}
    creator['https://dataverse.org/schema/citation/authorName'] = config.get('authorName', 'The Minority Report')
    creator['https://dataverse.org/schema/citation/authorAffiliation'] = config.get('authorAffiliation', 'United Nations Office for Disaster Risk Reduction')
    voc['http://purl.org/dc/terms/creator'] = creator

    # Contact
    contact = {}
    contact['https://dataverse.org/schema/citation/datasetContactEmail'] = config.get('datasetContactEmail', 'dummy@email.com')
    contact['https://dataverse.org/schema/citation/datasetContactName'] = config.get('datasetContactName', 'The Minority Report Team')
    voc['https://dataverse.org/schema/citation/datasetContact'] = contact

    # Description
    desc_text = croissant_data.get('description', '')
    if desc_text:
        desc = {}
        desc['https://dataverse.org/schema/citation/dsDescriptionValue'] = desc_text
        voc['https://dataverse.org/schema/citation/dsDescription'] = desc

    # License
    voc['http://schema.org/license'] = config.get('license_url', 'http://creativecommons.org/licenses/by/4.0')
    voc['http://purl.org/dc/terms/license'] = config.get('license_name', 'CC BY 4.0')
    voc['http://purl.org/dc/terms/rights'] = config.get('license_rights', 'Creative Commons Attribution 4.0 International')

    # Keywords (From Translations)
    # Mapping sc:alternateName to keywords
    keywords = []
    
    # Add source term itself
    source_term = croissant_data.get('name')
    if source_term:
         keywords.append({'https://dataverse.org/schema/citation/keywordValue': source_term})

    alternate_names = croissant_data.get('sc:alternateName', [])
    for alt in alternate_names:
        if isinstance(alt, dict) and '@value' in alt:
            kw_val = alt['@value']
            kw_lang = alt.get('@language', '')
            
            keyword_entry = {'https://dataverse.org/schema/citation/keywordValue': kw_val}
            if kw_lang:
                keyword_entry['https://dataverse.org/schema/citation/keywordVocabulary'] = kw_lang
                # Default to ISO list, but override if provided (e.g., from Wikidata/SparqlMuse)
                default_uri = 'https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes'
                keyword_entry['https://dataverse.org/schema/citation/keywordVocabularyURI'] = config.get('keyword_vocabulary_uri', default_uri)
            
            keywords.append(keyword_entry)
    
    if keywords:
        voc['https://dataverse.org/schema/citation/keyword'] = keywords

    return voc

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Convert Croissant JSON to Dataverse JSON-LD.")
    parser.add_argument("input_file", nargs='?', help="Path to input Croissant JSON file")
    
    args = parser.parse_args()
    
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                data = json.load(f)
            dv_json = croissant_to_dataverse(data)
            print(json.dumps(dv_json, indent=2))
        except FileNotFoundError:
            print(f"Error: File not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {args.input_file}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
