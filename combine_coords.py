#!/usr/bin/env python3
"""Combine coordinate files from segmented video processing."""

import argparse
from pathlib import Path
import re

def parse_segment_filename(filename: str):
    """Parse segment filename like 'segment_0_635.mp4' or 'segment_0_635_yolo11_coords.txt'"""
    # Extract the numeric part
    match = re.search(r'segment_(\d+)_(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot parse segment numbers from filename: {filename}")
    start = int(match.group(1))
    end = int(match.group(2))
    return start, end

def combine_coordinate_files(coord_files, output_path: Path):
    """Combine multiple coordinate files into one with adjusted frame numbers."""
    combined_lines = []
    header_written = False
    
    for coord_file in sorted(coord_files):
        coord_path = Path(coord_file)
        if not coord_path.exists():
            print(f"Warning: Coordinate file not found: {coord_path}")
            continue
            
        # Parse start frame from filename
        try:
            start_frame, _ = parse_segment_filename(coord_path.name)
        except ValueError as e:
            print(f"Warning: {e}, skipping {coord_path}")
            continue
            
        print(f"Processing {coord_path.name} (offset: +{start_frame})")
        
        with open(coord_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.rstrip('\n')  # Keep leading/trailing spaces, just remove newline
            if not line:
                continue
                
            # Handle header line (starts with #)
            if line.startswith('#'):
                if not header_written:
                    combined_lines.append(line)
                    header_written = True
                continue
                
            # Parse data line
            parts = line.split('\t')
            if len(parts) != 7:
                print(f"Warning: Malformed line in {coord_path.name} line {i+1}: {line}")
                continue
                
            # Adjust frame number
            try:
                original_frame = int(parts[0])
                adjusted_frame = original_frame + start_frame
                parts[0] = str(adjusted_frame)
                combined_lines.append('\t'.join(parts))
            except ValueError:
                print(f"Warning: Invalid frame number in {coord_path.name} line {i+1}: {parts[0]}")
                continue
    
    # Write combined file
    with open(output_path, 'w') as f:
        f.write('\n'.join(combined_lines))
    
    print(f"Combined {len(coord_files)} coordinate files into {output_path}")
    print(f"Total frames: {len(combined_lines) - (1 if header_written else 0)}")

def main():
    parser = argparse.ArgumentParser(description="Combine coordinate files from segmented video processing.")
    parser.add_argument("coord_files", nargs="+", help="Coordinate files to combine")
    parser.add_argument("-o", "--output", default="combined_coords.txt", help="Output file path")
    args = parser.parse_args()
    
    combine_coordinate_files(args.coord_files, Path(args.output))

if __name__ == "__main__":
    main()
