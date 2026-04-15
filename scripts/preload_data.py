#!/usr/bin/env python3
"""Preload HDF5 data to system cache before training.

This script reads all HDF5 files to load them into the system page cache,
which can significantly improve training speed by reducing disk I/O.

Usage:
    python scripts/preload_data.py --data_dir /path/to/data
    python scripts/preload_data.py --config configs/single_mol/geometric.yaml
"""
import argparse
import glob
import os
import sys
import time

import yaml
from tqdm import tqdm


def preload_file(filepath: str, verbose: bool = True) -> int:
    """Preload a single HDF5 file into memory cache.
    
    Args:
        filepath: Path to HDF5 file
        verbose: Print progress
        
    Returns:
        Number of bytes read
    """
    try:
        file_size = os.path.getsize(filepath)
        
        # Method 1: Read file in chunks (works for any file)
        chunk_size = 64 * 1024 * 1024  # 64 MB chunks
        bytes_read = 0
        
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
        
        if verbose:
            print(f"✓ Preloaded {filepath} ({file_size / 1e9:.2f} GB)")
        
        return bytes_read
    except Exception as e:
        print(f"✗ Failed to preload {filepath}: {e}")
        return 0


def get_files_from_config(config_path: str) -> list:
    """Extract HDF5 file paths from config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        List of HDF5 file paths
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    files = []
    
    # Get train_h5 paths
    if 'train_h5' in config and config['train_h5']:
        for pattern in config['train_h5']:
            files.extend(glob.glob(pattern))
    
    # Get val_h5 paths
    if 'val_h5' in config and config['val_h5']:
        for pattern in config['val_h5']:
            files.extend(glob.glob(pattern))
    
    return sorted(set(files))


def preload_data(file_paths: list, num_workers: int = 1) -> None:
    """Preload multiple HDF5 files into memory cache.
    
    Args:
        file_paths: List of file paths to preload
        num_workers: Number of parallel workers (not implemented yet)
    """
    if not file_paths:
        print("No files to preload!")
        return

    if num_workers != 1:
        print(f"[warn] num_workers={num_workers} is not implemented yet; using sequential preload.")
    
    print(f"\n{'='*70}")
    print(f"Preloading {len(file_paths)} files into system cache...")
    print(f"{'='*70}\n")
    
    total_bytes = 0
    start_time = time.time()
    
    for filepath in tqdm(file_paths, desc="Preloading files"):
        bytes_read = preload_file(filepath, verbose=False)
        total_bytes += bytes_read
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ Preloading complete!")
    print(f"   Total data: {total_bytes / 1e9:.2f} GB")
    print(f"   Time taken: {elapsed:.1f} seconds")
    print(f"   Speed: {total_bytes / elapsed / 1e6:.1f} MB/s")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preload HDF5 data to system cache"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Directory containing HDF5 files (will load all *.h5)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        help='Glob pattern for HDF5 files (e.g., "/path/to/data/**/*.h5")'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (not implemented yet)'
    )
    
    args = parser.parse_args()
    
    # Collect file paths
    file_paths = []
    
    if args.config:
        print(f"Loading file paths from config: {args.config}")
        file_paths.extend(get_files_from_config(args.config))
    
    if args.data_dir:
        print(f"Loading files from directory: {args.data_dir}")
        pattern = os.path.join(args.data_dir, "**/*.h5")
        file_paths.extend(glob.glob(pattern, recursive=True))
    
    if args.pattern:
        print(f"Loading files matching pattern: {args.pattern}")
        file_paths.extend(glob.glob(args.pattern, recursive=True))
    
    if not file_paths:
        print("Error: No files specified!")
        print("Use --config, --data_dir, or --pattern to specify files")
        sys.exit(1)
    
    # Remove duplicates and sort
    file_paths = sorted(set(file_paths))
    
    # Preload
    preload_data(file_paths, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
