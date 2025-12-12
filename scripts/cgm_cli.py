#!/usr/bin/env python3
"""CGM CLI launcher script.

This is a convenience script that can be run directly from the scripts directory.
The actual implementation is in cgm_format.cgm_cli for proper package integration.

Usage:
    python scripts/cgm_cli.py <command> [options]
    
Or install the package and use:
    cgm-cli <command> [options]
    python -m cgm_format.cgm_cli <command> [options]
"""

from cgm_format.cgm_cli import main

if __name__ == "__main__":
    main()
