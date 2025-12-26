# Documentation Structure

## Overview

The Pale Fire documentation has been organized into the `docs/` folder for better maintainability and navigation.

## Directory Structure

```
palefire/
├── README.md                      # Main project README
├── api.py                         # FastAPI REST API
├── palefire-cli.py                # Command-line interface
├── config.py                      # Configuration module
├── requirements.txt               # Python dependencies
├── env.example                    # Environment configuration template
│
├── modules/                       # Core modules
│   ├── __init__.py
│   └── PaleFireCore.py           # EntityEnricher, QuestionTypeDetector
│
├── docs/                          # 📚 All documentation
│   ├── README.md                 # Documentation index
│   │
│   ├── Getting Started/
│   │   ├── PALEFIRE_SETUP.md
│   │   ├── QUICK_REFERENCE.md
│   │   └── CONFIGURATION.md
│   │
│   ├── User Guides/
│   │   ├── CLI_GUIDE.md
│   │   └── API_GUIDE.md
│   │
│   ├── Features/
│   │   ├── RANKING_SYSTEM.md
│   │   ├── NER_ENRICHMENT.md
│   │   ├── QUESTION_TYPE_DETECTION.md
│   │   └── QUERY_MATCH_SCORING.md
│   │
│   ├── Advanced/
│   │   ├── ARCHITECTURE.md
│   │   ├── DATABASE_CLEANUP.md
│   │   ├── EXPORT_FEATURE.md
│   │   └── ENTITY_TYPES_UPDATE.md
│   │
│   └── Changelog/
│       ├── CHANGELOG_CONFIG.md
│       ├── MIGRATION_SUMMARY.md
│       └── EXPORT_CHANGES.md
│
├── example_episodes.json          # Example data
└── example_export.json            # Example export
```

## Documentation Categories

### 1. Getting Started (3 docs)
Essential documentation for new users:
- Setup and installation
- Quick command reference
- Configuration options

### 2. User Guides (2 docs)
Detailed usage guides:
- CLI commands and examples
- REST API endpoints and usage

### 3. Features (4 docs)
Feature-specific documentation:
- Ranking algorithm details
- NER enrichment process
- Question type detection
- Query matching scoring

### 4. Advanced (4 docs)
Advanced topics and maintenance:
- System architecture
- Database operations
- Export functionality
- Entity type system

### 5. Changelog (3 docs)
Version history and migrations:
- Configuration changes
- Migration summaries
- Format updates

## Navigation

### From Root
- Start at `README.md`
- Navigate to `docs/README.md` for full index
- Access any doc via `docs/FILENAME.md`

### Within Docs
- Use relative links: `[Text](FILENAME.md)`
- All cross-references work within docs folder
- No path prefixes needed

### External References
- From root to docs: `docs/FILENAME.md`
- From docs to root: `../FILENAME`

## File Naming Convention

- **UPPERCASE_SNAKE_CASE.md** - All documentation files
- **Descriptive names** - Clear purpose from filename
- **Consistent suffixes**:
  - `_GUIDE.md` - User guides
  - `_SYSTEM.md` - System documentation
  - `_FEATURE.md` - Feature documentation
  - `_CHANGES.md` - Changelog entries

## Link Patterns

### Internal Links (within docs/)
```markdown
[CLI Guide](CLI_GUIDE.md)
[Configuration](CONFIGURATION.md)
```

### External Links (from docs/ to root)
```markdown
[Main README](../README.md)
[API Code](../api.py)
```

### Links from Root to Docs
```markdown
[Setup Guide](docs/PALEFIRE_SETUP.md)
[Documentation Index](docs/README.md)
```

## Maintenance

### Adding New Documentation

1. Create file in `docs/` folder
2. Use UPPERCASE_SNAKE_CASE naming
3. Add entry to `docs/README.md`
4. Update relevant cross-references
5. Add to appropriate category

### Updating Documentation

1. Edit file in `docs/` folder
2. Update modification date
3. Check cross-references still work
4. Update `docs/README.md` if needed

### Removing Documentation

1. Remove file from `docs/` folder
2. Remove entry from `docs/README.md`
3. Update any cross-references
4. Archive if needed (move to `docs/archive/`)

## Benefits of This Structure

### ✅ Organization
- All docs in one place
- Clear categorization
- Easy to find information

### ✅ Maintainability
- Simple to add new docs
- Easy to update cross-references
- Clear structure for contributors

### ✅ Navigation
- Centralized index
- Logical grouping
- Quick access to any doc

### ✅ Scalability
- Room for growth
- Flexible categorization
- Can add subdirectories

## Documentation Standards

### Format
- Markdown (.md) format
- UTF-8 encoding
- Unix line endings (LF)

### Structure
- Title (# heading)
- Overview section
- Table of contents (if long)
- Main content
- See Also section
- Footer with version/date

### Style
- Clear, concise language
- Code examples with syntax highlighting
- Screenshots where helpful
- Consistent formatting

### Cross-References
- Use relative links
- Descriptive link text
- Check links work
- Update when files move

## Migration Notes

### What Changed
- Moved all `.md` files to `docs/` folder
- Created `docs/README.md` index
- Updated main `README.md` references
- Preserved all content

### What Stayed the Same
- File names unchanged
- Content unchanged
- Cross-references within docs work
- All information preserved

### Breaking Changes
- ❌ Old links from root broken (e.g., `CLI_GUIDE.md`)
- ✅ New links work (e.g., `docs/CLI_GUIDE.md`)
- ✅ Links within docs still work

## Quick Reference

### Common Tasks

**Find documentation:**
```bash
ls docs/
```

**Search documentation:**
```bash
grep -r "search term" docs/
```

**View documentation index:**
```bash
cat docs/README.md
```

**Access specific doc:**
```bash
# From root
cat docs/CLI_GUIDE.md

# From docs
cd docs && cat CLI_GUIDE.md
```

## See Also

- [Documentation Index](README.md) - Complete documentation list
- [Main README](../README.md) - Project overview
- [Configuration](CONFIGURATION.md) - Configuration options

---

**Documentation Structure v1.0** - Organized and Accessible! 📁

