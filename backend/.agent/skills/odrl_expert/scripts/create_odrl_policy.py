import json
import argparse
import sys
import os

def create_odrl_policy(uid=None, target=None, action=None, assigner=None, duty_action=None, duty_desc=None):
    """
    Creates an ODRL policy based on a template.
    """
    policy = {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "@type": "Set",
        "uid": uid or "did:oyd:zQmbtKvHhpjrZ4s3PjihtUdD9xZ5RnqPTiBb6mLCaonXriK",
        "permission": [
            {
                "target": target or "https://github.com/TheWorldAvatar/ontoclimateadapt",
                "action": action or "odrl:use",
                "assigner": assigner or "https://github.com/TheWorldAvatar"
            },
            {
                "target": target or "https://github.com/TheWorldAvatar/ontoclimateadapt",
                "action": "odrl:reproduce"
            },
            {
                "target": target or "https://github.com/TheWorldAvatar/ontoclimateadapt",
                "action": "odrl:distribute"
            }
        ],
        "duty": [
            {
                "action": duty_action or "odrl:attribute",
                "description": duty_desc or "Attribution to TheWorldAvatar / Cambridge CARES"
            }
        ]
    }
    return policy

def main():
    parser = argparse.ArgumentParser(description="Create an ODRL policy JSON-LD.")
    parser.add_argument("--uid", help="Unique identifier (e.g., DID)")
    parser.add_argument("--target", help="Target resource URL")
    parser.add_argument("--assigner", help="Assigner URL")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    policy = create_odrl_policy(
        uid=args.uid,
        target=args.target,
        assigner=args.assigner
    )
    
    output_json = json.dumps(policy, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"ODRL policy saved to {args.output}")
    else:
        print(output_json)

if __name__ == "__main__":
    main()
