# Scripts

Helper scripts for maintaining the cgm_format project.

## regenerate_all_schemas.py

Automatically regenerates all JSON schema files from their Python schema definitions.

### Usage

```bash
# As executable (recommended - uses uv automatically)
./scripts/regenerate_all_schemas.py

# Or using uv explicitly
uv run python scripts/regenerate_all_schemas.py

# Or directly with Python (requires dependencies installed)
python scripts/regenerate_all_schemas.py
```

### What it does

1. Discovers all format modules in `src/cgm_format/formats/` (excluding `__init__.py` and `*_WIP.py`)
2. Dynamically imports each module
3. Calls `regenerate_schema_json()` function if it exists
4. Generates/updates corresponding `.json` schema files

### Output

The script regenerates:
- `dexcom.json` - Dexcom G6/G7 format schema
- `libre.json` - FreeStyle Libre 3 format schema
- `unified.json` - Unified CGM format schema

### When to run

Run this script after:
- Modifying enum values in format definitions
- Adding/removing columns in schemas
- Changing column descriptions or constraints
- Any other schema-related changes

This ensures the JSON schema files stay in sync with the Python schema definitions.

