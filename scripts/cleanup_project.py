"""
Project cleanup script - removes unnecessary files and folders.
"""
import os
import shutil
from pathlib import Path

def remove_empty_dirs(path: Path):
    """Remove empty directories."""
    if path.exists() and path.is_dir():
        try:
            if not any(path.iterdir()):
                path.rmdir()
                print(f"  Removed empty directory: {path}")
                return True
        except Exception as e:
            print(f"  Could not remove {path}: {e}")
    return False

def cleanup():
    """Main cleanup function."""
    print("=" * 80)
    print("PROJECT CLEANUP")
    print("=" * 80)
    
    root = Path(".")
    
    # 1. Remove empty directories
    print("\n1. Removing empty directories...")
    empty_dirs = [
        "eval_outputs",
    ]
    for dir_name in empty_dirs:
        dir_path = root / dir_name
        remove_empty_dirs(dir_path)
    
    # 2. Remove __pycache__ directories (will be regenerated)
    print("\n2. Removing __pycache__ directories...")
    pycache_dirs = list(root.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            # Skip venv __pycache__ (too many files)
            if "venv" in str(pycache_dir):
                continue
            shutil.rmtree(pycache_dir)
            print(f"  Removed: {pycache_dir}")
        except Exception as e:
            print(f"  Could not remove {pycache_dir}: {e}")
    
    # 3. Remove obsolete result files
    print("\n3. Removing obsolete result files...")
    results_dir = root / "results"
    obsolete_files = [
        "seed_50_fixed.jsonl",
        "seed_50_fixed2.jsonl",
        "seed_50_cold_with_scores.jsonl",
        "error_taxonomy_table.txt",  # Replaced by phase1_1_error_taxonomy_final.md
        "phase1_1_error_taxonomy.md",  # Replaced by phase1_1_error_taxonomy_final.md
    ]
    for file_name in obsolete_files:
        file_path = results_dir / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  Removed: {file_path}")
            except Exception as e:
                print(f"  Could not remove {file_path}: {e}")
    
    # 4. Remove obsolete test files (they import non-existent modules)
    print("\n4. Removing obsolete test files...")
    obsolete_tests = [
        "test_factcheck_regression.py",
        "test_population_regression.py",
    ]
    for test_file in obsolete_tests:
        test_path = root / test_file
        if test_path.exists():
            try:
                test_path.unlink()
                print(f"  Removed: {test_path}")
            except Exception as e:
                print(f"  Could not remove {test_path}: {e}")
    
    # 5. Remove old ZIP archives (if not needed)
    print("\n5. Checking old ZIP archives...")
    zip_files = [
        "app.v1.zip",
        "app.v2.zip",
        "app.v3.zip",
    ]
    for zip_file in zip_files:
        zip_path = root / zip_file
        if zip_path.exists():
            print(f"  Found: {zip_path} (keeping for now - remove manually if not needed)")
    
    # 6. Remove duplicate smoke_30.jsonl in scripts/eval (already in data/eval)
    print("\n6. Checking for duplicate files...")
    duplicate_file = root / "scripts" / "eval" / "smoke_30.jsonl"
    if duplicate_file.exists():
        try:
            duplicate_file.unlink()
            print(f"  Removed duplicate: {duplicate_file}")
        except Exception as e:
            print(f"  Could not remove {duplicate_file}: {e}")
    
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)
    print("\nNote: The following were kept for documentation purposes:")
    print("  - archive/ (historical pipeline versions)")
    print("  - *.md files in root (documentation)")
    print("  - app.v*.zip (version archives - remove manually if not needed)")
    print()

if __name__ == "__main__":
    cleanup()
