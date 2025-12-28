# Prompts Directory

This directory contains all prompts used for AI/LLM interactions in the Pale Fire project.

## Purpose

Store all prompt templates, system prompts, and instruction sets used for:
- LLM queries and interactions
- Knowledge graph generation
- Entity extraction instructions
- Question type detection prompts
- Any other AI-related prompts

## Structure

Organize prompts by category:

```
prompts/
├── README.md              # This file
├── system/                # System prompts
├── queries/               # Query-related prompts
├── extraction/            # Entity/keyword extraction prompts
└── templates/             # Reusable prompt templates
```

## Usage

When creating new prompts:

1. **Name files clearly**: Use descriptive names like `entity_extraction_prompt.md` or `question_detection_system.txt`
2. **Include metadata**: Add comments or headers describing:
   - Purpose of the prompt
   - Target model/use case
   - Version or date
   - Example usage
3. **Version control**: Keep prompts in version control for reproducibility
4. **Documentation**: Update this README when adding new prompt categories

## Examples

### System Prompt Example
```markdown
# Entity Extraction System Prompt

**Purpose**: Extract named entities from text
**Model**: spaCy / Pattern-based
**Version**: 1.0

[Prompt content here...]
```

### Template Example
```markdown
# Query Template

**Purpose**: Generate knowledge graph queries
**Variables**: {query_type}, {entity_types}, {temporal_range}

[Template content here...]
```

## Best Practices

1. ✅ Keep prompts modular and reusable
2. ✅ Document prompt parameters and variables
3. ✅ Include examples of expected input/output
4. ✅ Version prompts when making significant changes
5. ✅ Test prompts before committing
6. ✅ Reference prompts in code with clear paths

