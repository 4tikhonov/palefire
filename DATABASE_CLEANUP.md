# Database Cleanup Guide

## Overview

The `clean` command allows you to clear all data from the Neo4j database, providing a fresh start for testing or removing old data.

## Usage

### Basic Cleanup (with confirmation)

```bash
python palefire-cli.py clean
```

This will:
1. Show current database statistics (nodes and relationships)
2. Prompt for confirmation
3. Delete all nodes and relationships
4. Show cleanup results

### Quick Cleanup (skip confirmation)

```bash
python palefire-cli.py clean --confirm
```

⚠️ **Warning:** This skips the confirmation prompt and immediately deletes all data!

### Nodes-Only Cleanup

```bash
python palefire-cli.py clean --nodes-only
```

This deletes only nodes and relationships while preserving:
- Database indexes
- Constraints
- Schema definitions

## Command Options

| Option | Description |
|--------|-------------|
| `--confirm` | Skip confirmation prompt and clean immediately |
| `--nodes-only` | Delete only nodes (keep database structure) |

## Examples

### Example 1: Interactive Cleanup

```bash
$ python palefire-cli.py clean

================================================================================
🗑️  DATABASE CLEANUP
================================================================================
Current database contents:
  Nodes: 1523
  Relationships: 4567

⚠️  WARNING: This will permanently delete all data!
   Mode: Complete cleanup (all nodes, relationships, and data)

Are you sure you want to continue? (yes/no): yes

🔄 Cleaning database...

================================================================================
✅ DATABASE CLEANED SUCCESSFULLY
================================================================================
Deleted:
  Nodes: 1523
  Relationships: 4567

The database is now empty and ready for new data.
================================================================================
```

### Example 2: Automated Cleanup

```bash
# For scripts or automation
python palefire-cli.py clean --confirm
```

### Example 3: Cancel Cleanup

```bash
$ python palefire-cli.py clean

⚠️  WARNING: This will permanently delete all data!
Are you sure you want to continue? (yes/no): no

❌ Cleanup cancelled.
```

### Example 4: Clean and Re-ingest

```bash
# Clean database
python palefire-cli.py clean --confirm

# Ingest fresh data
python palefire-cli.py ingest --file new_data.json

# Query the new data
python palefire-cli.py query "Your question?"
```

## Use Cases

### 1. Testing and Development

```bash
# Clean before each test run
python palefire-cli.py clean --confirm
python palefire-cli.py ingest --demo
# Run tests...
```

### 2. Data Migration

```bash
# Remove old data
python palefire-cli.py clean --confirm

# Import new data format
python palefire-cli.py ingest --file new_format.json
```

### 3. Removing Corrupted Data

```bash
# Clean corrupted database
python palefire-cli.py clean --confirm

# Re-ingest from backup
python palefire-cli.py ingest --file backup.json
```

### 4. Starting Fresh

```bash
# Clear everything and start over
python palefire-cli.py clean --confirm
python palefire-cli.py ingest --file episodes.json
```

## Safety Features

### 1. Confirmation Prompt

By default, the command asks for confirmation:
```
Are you sure you want to continue? (yes/no):
```

Only "yes" or "y" will proceed with cleanup.

### 2. Statistics Display

Shows what will be deleted:
```
Current database contents:
  Nodes: 1523
  Relationships: 4567
```

### 3. Verification

After cleanup, verifies the database is empty:
```
Deleted:
  Nodes: 1523
  Relationships: 4567
```

### 4. Empty Database Detection

If the database is already empty:
```
✅ Database is already empty!
```

## What Gets Deleted

### Standard Cleanup (`clean`)

Deletes:
- ✅ All nodes
- ✅ All relationships
- ✅ All node properties
- ✅ All relationship properties

Preserves:
- ✅ Database structure
- ✅ Indexes
- ✅ Constraints

### Nodes-Only Cleanup (`clean --nodes-only`)

Same as standard cleanup (currently identical behavior).

## Best Practices

### 1. Always Backup First

```bash
# Export current data before cleaning
python palefire-cli.py query "..." --export backup.json

# Then clean
python palefire-cli.py clean
```

### 2. Use Confirmation in Production

```bash
# Good: Requires confirmation
python palefire-cli.py clean

# Risky: No confirmation
python palefire-cli.py clean --confirm
```

### 3. Verify After Cleanup

```bash
# Clean database
python palefire-cli.py clean --confirm

# Verify it's empty (should return no results)
python palefire-cli.py query "test"
```

### 4. Document Your Cleanup

```bash
# Add to your scripts
echo "Cleaning database at $(date)" >> cleanup.log
python palefire-cli.py clean --confirm
echo "Cleanup completed at $(date)" >> cleanup.log
```

## Troubleshooting

### Problem: Cleanup Fails

**Symptoms:**
```
❌ Error cleaning database: ...
```

**Solutions:**
1. Check Neo4j is running
2. Verify connection credentials
3. Check database permissions
4. Look for locked nodes

### Problem: Some Nodes Remain

**Symptoms:**
```
⚠️  CLEANUP INCOMPLETE
Remaining nodes: 5
```

**Solutions:**
1. Run cleanup again
2. Check for constraint violations
3. Manually delete remaining nodes:
   ```cypher
   MATCH (n) DETACH DELETE n
   ```

### Problem: Permission Denied

**Symptoms:**
```
Error: Permission denied
```

**Solutions:**
1. Check Neo4j user permissions
2. Ensure user has DELETE privileges
3. Use admin credentials

## Advanced Usage

### Scripted Cleanup

```bash
#!/bin/bash
# cleanup_and_reingest.sh

echo "Starting cleanup process..."

# Clean database
python palefire-cli.py clean --confirm

if [ $? -eq 0 ]; then
    echo "Cleanup successful, starting ingestion..."
    python palefire-cli.py ingest --file data.json
else
    echo "Cleanup failed, aborting."
    exit 1
fi
```

### Conditional Cleanup

```python
import subprocess
import sys

def clean_if_needed():
    """Clean database if it has more than 10000 nodes."""
    # Check node count
    result = subprocess.run(
        ['python', 'palefire-cli.py', 'query', 'MATCH (n) RETURN count(n)'],
        capture_output=True
    )
    
    # If too many nodes, clean
    if node_count > 10000:
        subprocess.run(['python', 'palefire-cli.py', 'clean', '--confirm'])
```

### Backup Before Clean

```bash
#!/bin/bash
# safe_clean.sh

# Create backup
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).json"
python palefire-cli.py query "..." --export "$BACKUP_FILE"

# Clean database
python palefire-cli.py clean --confirm

echo "Backup saved to: $BACKUP_FILE"
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Clean Test Database
  run: |
    python palefire-cli.py clean --confirm
    python palefire-cli.py ingest --file test_data.json
```

### Jenkins

```groovy
stage('Clean Database') {
    steps {
        sh 'python palefire-cli.py clean --confirm'
        sh 'python palefire-cli.py ingest --file ${TEST_DATA}'
    }
}
```

### Docker

```dockerfile
# Clean database on container start
CMD ["sh", "-c", "python palefire-cli.py clean --confirm && python palefire-cli.py ingest --demo"]
```

## FAQ

### Q: Can I undo a cleanup?

**A:** No, cleanup is permanent. Always backup first!

### Q: Does cleanup affect other databases?

**A:** No, it only affects the configured Neo4j database.

### Q: How long does cleanup take?

**A:** Depends on database size:
- Small (< 1000 nodes): < 1 second
- Medium (1000-10000 nodes): 1-5 seconds
- Large (> 10000 nodes): 5-30 seconds

### Q: Can I clean specific nodes?

**A:** No, the clean command removes all nodes. For selective deletion, use Cypher queries directly.

### Q: What happens to indexes and constraints?

**A:** They are preserved by default. Use `--nodes-only` to ensure this.

## See Also

- [CLI Guide](CLI_GUIDE.md) - Complete CLI documentation
- [Quick Reference](QUICK_REFERENCE.md) - Quick command reference
- [Configuration](CONFIGURATION.md) - Configuration options

---

**Database Cleanup v1.0** - Clean Slate, Fresh Start! 🗑️

