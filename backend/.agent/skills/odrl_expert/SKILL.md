---
name: odrl_expert
description: Resolve Decentralized Identifiers (DIDs) from ODRL (Open Digital Rights Language) files and fetch associated resource metadata (URL, name, description) using the Universal Resolver.
---

# ODRL Expert Skill

The ODRL Expert skill allows the agent to read ODRL files and resolve DIDs to their corresponding resource metadata. This skill is particularly useful for understanding the rights and associated resources described in ODRL documents.

## Tools

### 1. Resolve DID from ODRL
Reads an ODRL file (typically `.odrlmd`), extracts the DID, and resolves it using the Universal Resolver (`https://dev.uniresolver.io`).

**Usage:**
```bash
python3 backend/.agent/skills/odrl_expert/scripts/resolve_odrl_did.py <ODRL_FILE_PATH_OR_CONTENT>
```

**Example:**
`python3 backend/.agent/skills/odrl_expert/scripts/resolve_odrl_did.py /path/to/resource/ODRL.md`

### 2. GitHub ODRL Resolver
Lists `ODRL.md` files from a GitHub repository or directory and resolves the DIDs found within each one.

**Usage:**
```bash
python3 backend/.agent/skills/odrl_expert/scripts/github_odrl_resolver.py <GITHUB_REPO_OR_DIR_URL>
```

**Example:**
`python3 backend/.agent/skills/odrl_expert/scripts/github_odrl_resolver.py https://github.com/CA4EOSC/ODRL/tree/main/catalogue`

### 3. ODRL Policy Creation
Generates a standard ODRL policy JSON-LD based on the TheWorldAvatar template.

**Usage:**
```bash
python3 backend/.agent/skills/odrl_expert/scripts/create_odrl_policy.py --uid <DID_OR_UID> --target <TARGET_URL>
```

**Example:**
`python3 backend/.agent/skills/odrl_expert/scripts/create_odrl_policy.py --uid "did:oyd:zQmbtKvHhpjrZ4s3PjihtUdD9xZ5RnqPTiBb6mLCaonXriK" --target "https://github.com/TheWorldAvatar/ontoclimateadapt"`
