#!/usr/bin/env -S uv run python
"""Regenerate all schema JSON files in the formats folder.

This script dynamically discovers all format modules with regenerate_schema_json()
functions and calls them to regenerate their corresponding JSON schema files.

Usage:
    python scripts/regenerate_all_schemas.py
    
    Or with uv:
    uv run scripts/regenerate_all_schemas.py
"""

import sys
from pathlib import Path
from importlib import import_module
from typing import Callable

# Add src to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def discover_format_modules() -> list[tuple[str, Path]]:
    """Discover all Python modules in the formats folder.
    
    Returns:
        List of tuples (module_name, module_path) for format modules
    """
    formats_dir = project_root / "src" / "cgm_format" / "formats"
    
    if not formats_dir.exists():
        raise FileNotFoundError(f"Formats directory not found: {formats_dir}")
    
    # Find all .py files, excluding __init__.py, __pycache__, and *_WIP.py
    format_modules = []
    for py_file in formats_dir.glob("*.py"):
        if py_file.name == "__init__.py" or py_file.name.endswith("_WIP.py"):
            continue
        
        module_name = py_file.stem  # e.g., "unified", "dexcom", "libre"
        format_modules.append((module_name, py_file))
    
    return sorted(format_modules)  # Sort for consistent ordering


def regenerate_schema(module_name: str, module_path: Path) -> bool:
    """Attempt to import a module and call its regenerate_schema_json() function.
    
    Args:
        module_name: Name of the module (e.g., "unified")
        module_path: Path to the module file
        
    Returns:
        True if schema was regenerated successfully, False otherwise
    """
    try:
        # Import the module dynamically
        full_module_name = f"cgm_format.formats.{module_name}"
        module = import_module(full_module_name)
        
        # Check if it has a regenerate_schema_json function
        if not hasattr(module, "regenerate_schema_json"):
            print(f"⊘ Skipping {module_name}: no regenerate_schema_json() function")
            return False
        
        # Call the regenerate function
        regenerate_func: Callable[[], None] = getattr(module, "regenerate_schema_json")
        regenerate_func()
        
        return True
        
    except Exception as e:
        print(f"✗ Error regenerating {module_name}: {type(e).__name__}: {e}")
        return False


def main() -> None:
    """Main entry point: regenerate all schemas."""
    print("🔄 Regenerating all schema JSON files...\n")
    
    # Discover all format modules
    format_modules = discover_format_modules()
    
    if not format_modules:
        print("⚠️  No format modules found in formats directory")
        sys.exit(1)
    
    print(f"Discovered {len(format_modules)} format module(s): {', '.join(name for name, _ in format_modules)}\n")
    
    # Regenerate each schema
    successes = 0
    failures = 0
    
    for module_name, module_path in format_modules:
        result = regenerate_schema(module_name, module_path)
        if result:
            successes += 1
        else:
            failures += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  ✓ Successfully regenerated: {successes}")
    if failures > 0:
        print(f"  ✗ Failed: {failures}")
    print(f"{'='*60}\n")
    
    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

