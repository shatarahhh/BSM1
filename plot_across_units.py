#!/usr/bin/env python3
"""
Standalone script to plot across-units data from saved NPZ files.
Usage: python plot_across_units.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ============================================================================
# HARDCODED PATHS
# ============================================================================
DATA_DIR = 'results_takacs/across_units_data'
OUTPUT_DIR = 'results_takacs/across_units'

# Component names in the expected order
COMPONENT_NAMES = ['S_I', 'S_S', 'X_I', 'X_S', 'X_H', 'X_A', 
                   'X_P', 'S_O', 'S_NO', 'S_NH', 'S_ND', 'X_ND', 'S_ALK']

# Particulate components (by index in the component list)
PARTICULATE_IDX = [2, 3, 4, 5, 6, 11]  # X_I, X_S, X_H, X_A, X_P, X_ND

# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plot_component(comp_name, comp_idx, data, output_dir):
    """
    Plot a single component's across-units profile.
    
    Args:
        comp_name: Name of the component (e.g., 'S_I')
        comp_idx: Index of the component in the component list
        data: Dictionary containing the loaded NPZ data
        output_dir: Directory to save the output plot
    """
    fig, axes = plt.subplots(8, 1, figsize=(12, 18), sharex=True)
    
    t_span = data['t_days']
    
    # 1) Influent
    axes[0].plot(t_span, data['influent'])
    axes[0].set_title(f'Influent → {comp_name}')
    
    # 2-6) After Tanks 1-5
    for tank in range(5):
        tank_key = f'tank{tank + 1}'
        axes[tank + 1].plot(t_span, data[tank_key])
        axes[tank + 1].set_title(f'After Tank {tank + 1} → {comp_name}')
    
    # 7) Clarifier Effluent
    axes[6].plot(t_span, data['effluent'])
    axes[6].set_title(f'Clarifier Effluent → {comp_name}')
    
    # 8) Clarifier RAS
    axes[7].set_title(f'Clarifier RAS → {comp_name}')
    
    # Check if RAS data is valid (not all NaNs)
    ras_data = data['ras']
    is_particulate = comp_idx in PARTICULATE_IDX
    has_valid_ras = not np.all(np.isnan(ras_data))
    
    if is_particulate or has_valid_ras:
        axes[7].plot(t_span, ras_data)
    else:
        axes[7].text(0.5, 0.5, 'RAS not plotted (no soluble transport detected)',
                    ha='center', va='center', transform=axes[7].transAxes)
    
    # Cosmetics
    for ax in axes:
        ax.grid(True)
        ax.set_ylabel('g/m³')
    axes[-1].set_xlabel('Time (days)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{comp_name}_across_units.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    """Main function to load data and create all plots."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from: {DATA_DIR}")
    print(f"Saving plots to: {OUTPUT_DIR}")
    print("-" * 60)
    
    # Check if manifest exists (optional, for information only)
    manifest_path = os.path.join(DATA_DIR, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"Found manifest with {len(manifest)} components")
    
    # Loop through all components
    for comp_idx, comp_name in enumerate(COMPONENT_NAMES):
        npz_path = os.path.join(DATA_DIR, f'{comp_name}.npz')
        
        if not os.path.exists(npz_path):
            print(f"WARNING: {npz_path} not found, skipping...")
            continue
        
        # Load the NPZ file
        data = np.load(npz_path)
        
        # Plot the component
        plot_component(comp_name, comp_idx, data, OUTPUT_DIR)
    
    print("-" * 60)
    print(f"All plots completed! Check {OUTPUT_DIR}")


if __name__ == '__main__':
    main()