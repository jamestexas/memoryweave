#!/usr/bin/env python3
"""
Migration script for MemoryWeave.

This script helps migrate from the legacy architecture to the component-based architecture.
It can scan Python files for deprecated imports and suggest replacements.
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

# Mapping of deprecated imports to their replacements
IMPORT_REPLACEMENTS = {
    r"from\s+memoryweave\.core\.contextual_fabric\s+import\s+(\w+)": 
        lambda match: f"from memoryweave.components.memory_manager import MemoryManager  # Migrated from core.contextual_fabric",
    
    r"from\s+memoryweave\.core\.contextual_memory\s+import\s+ContextualMemory":
        lambda match: "from memoryweave.components.memory_manager import MemoryManager  # Migrated from core.contextual_memory",
    
    r"from\s+memoryweave\.core\.memory_retriever\s+import\s+MemoryRetriever":
        lambda match: "from memoryweave.components.retriever import Retriever  # Migrated from core.memory_retriever",
    
    r"from\s+memoryweave\.core\.core_memory\s+import\s+CoreMemory":
        lambda match: "from memoryweave.storage.memory_store import MemoryStore  # Migrated from core.core_memory",
    
    r"from\s+memoryweave\.core\.category_manager\s+import\s+CategoryManager":
        lambda match: "from memoryweave.components.category_manager import CategoryManager  # Migrated from core.category_manager",
    
    r"from\s+memoryweave\.core\.memory_encoding\s+import\s+MemoryEncoder":
        lambda match: "from memoryweave.components.memory_adapter import MemoryAdapter  # Migrated from core.memory_encoding",
    
    r"from\s+memoryweave\.core\.refactored_retrieval\s+import\s+RefactoredRetriever":
        lambda match: "from memoryweave.components.retriever import Retriever  # Migrated from core.refactored_retrieval",
    
    r"import\s+memoryweave\.core\.contextual_memory":
        lambda match: "import memoryweave.components.memory_manager  # Migrated from core.contextual_memory",
    
    r"import\s+memoryweave\.core":
        lambda match: "# Migration needed: Replace 'import memoryweave.core' with specific imports from components",
}

# Mapping of class instantiations to their replacements
CLASS_REPLACEMENTS = {
    r"ContextualMemory\s*\(": {
        "replacement": "MemoryManager(",
        "message": "Replace ContextualMemory with MemoryManager (different parameters may be needed)"
    },
    r"CoreMemory\s*\(": {
        "replacement": "MemoryStore(",
        "message": "Replace CoreMemory with MemoryStore (different parameters may be needed)"
    },
    r"MemoryRetriever\s*\(": {
        "replacement": "Retriever(",
        "message": "Replace MemoryRetriever with Retriever (different parameters may be needed)"
    },
    r"RefactoredRetriever\s*\(": {
        "replacement": "Retriever(",
        "message": "Replace RefactoredRetriever with Retriever (different parameters may be needed)"
    },
}

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def scan_file(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Scan a Python file for deprecated imports and class instantiations.
    
    Returns:
        Tuple of (import_issues, class_issues)
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    import_issues = []
    for pattern, replacement_func in IMPORT_REPLACEMENTS.items():
        for match in re.finditer(pattern, content):
            import_issues.append({
                "line": content.count("\n", 0, match.start()) + 1,
                "match": match.group(0),
                "replacement": replacement_func(match)
            })
    
    class_issues = []
    for pattern, info in CLASS_REPLACEMENTS.items():
        for match in re.finditer(pattern, content):
            class_issues.append({
                "line": content.count("\n", 0, match.start()) + 1,
                "match": match.group(0),
                "replacement": info["replacement"],
                "message": info["message"]
            })
    
    return import_issues, class_issues

def main():
    parser = argparse.ArgumentParser(description="Scan and suggest migrations for MemoryWeave legacy code")
    parser.add_argument("directory", help="Directory to scan for Python files")
    parser.add_argument("--apply", action="store_true", help="Apply suggested changes (experimental)")
    args = parser.parse_args()
    
    python_files = find_python_files(args.directory)
    print(f"Found {len(python_files)} Python files to scan")
    
    total_import_issues = 0
    total_class_issues = 0
    
    for file_path in python_files:
        import_issues, class_issues = scan_file(file_path)
        
        if import_issues or class_issues:
            print(f"\n{file_path}:")
            
            if import_issues:
                print("  Import issues:")
                for issue in import_issues:
                    print(f"    Line {issue['line']}: {issue['match']}")
                    print(f"      Suggested replacement: {issue['replacement']}")
                total_import_issues += len(import_issues)
            
            if class_issues:
                print("  Class instantiation issues:")
                for issue in class_issues:
                    print(f"    Line {issue['line']}: {issue['match']}")
                    print(f"      Suggested replacement: {issue['replacement']}")
                    print(f"      Note: {issue['message']}")
                total_class_issues += len(class_issues)
    
    print(f"\nSummary: Found {total_import_issues} import issues and {total_class_issues} class instantiation issues")
    print("\nMigration guidance:")
    print("1. Replace imports from the core module with imports from components")
    print("2. Update class instantiations to use the new component classes")
    print("3. Note that parameters between old and new classes may differ")
    print("4. For complex migrations, use the adapters in memoryweave.adapters")
    print("5. See MIGRATION_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()