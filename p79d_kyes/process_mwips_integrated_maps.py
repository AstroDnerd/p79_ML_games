"""
MWIPS Integrated Maps Inspector
Understand the integrated map files and compare with individual cloud predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import json


def inspect_integrated_maps(integrated_dir):
    """
    Detailed inspection of integrated map files
    
    These are likely:
    - LVmap: Longitude-Velocity map (position-position integrated)
    - m0map: Moment 0 integrated intensity map (velocity integrated)
    """
    integrated_dir = Path(integrated_dir)
    
    for fits_file in integrated_dir.glob('*.fits'):
        print(f"FILE: {fits_file.name}")
        
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            
            print(f"\nDimensions: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Data range: [{np.nanmin(data):.3e}, {np.nanmax(data):.3e}]")
            print(f"Mean: {np.nanmean(data):.3e}")
            print(f"Std: {np.nanstd(data):.3e}")
            print(f"Non-zero pixels: {np.sum(data != 0):,} ({100*np.sum(data != 0)/data.size:.1f}%)")
            
            print("\nHeader Keywords:")
            for key in ['NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                       'CTYPE1', 'CTYPE2', 'CTYPE3', 
                       'CRVAL1', 'CRVAL2', 'CRVAL3',
                       'CDELT1', 'CDELT2', 'CDELT3',
                       'BUNIT', 'OBJECT']:
                if key in header:
                    print(f"  {key:8s}: {header[key]}")
            
            # Visualize
            visualize_integrated_map(data, fits_file.stem)


def visualize_integrated_map(data, title):
    """Create visualization of integrated map"""
    
    if data.ndim == 2:
        # 2D map
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Main map
        im = axes[0].imshow(data, origin='lower', cmap='viridis', aspect='auto')
        axes[0].set_title(f'{title}\nIntegrated Map', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X [pixels]')
        axes[0].set_ylabel('Y [pixels]')
        plt.colorbar(im, ax=axes[0], label='Intensity')
        
        # Histogram
        data_flat = data[data != 0].flatten()
        axes[1].hist(data_flat, bins=100, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Intensity')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Intensity Distribution', fontsize=12, fontweight='bold')
        axes[1].set_yscale('log')
        
    elif data.ndim == 3:
        # 3D cube - show slices
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Middle slices along each axis
        mid_z = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_x = data.shape[2] // 2
        
        im0 = axes[0, 0].imshow(data[mid_z, :, :], origin='lower', cmap='viridis')
        axes[0, 0].set_title(f'Slice at Z={mid_z}')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(data[:, mid_y, :], origin='lower', cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Slice at Y={mid_y}')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[1, 0].imshow(data[:, :, mid_x], origin='lower', cmap='viridis', aspect='auto')
        axes[1, 0].set_title(f'Slice at X={mid_x}')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Histogram
        data_flat = data[data != 0].flatten()
        axes[1, 1].hist(data_flat, bins=100, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Intensity Distribution')
        axes[1, 1].set_yscale('log')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'../data/mwips/{title}_inspection.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {title}_inspection.png")
    plt.close()


def compare_with_individual_clouds(integrated_map_file, predictions_json, clouds_dir):
    """
    Compare integrated map with sum of individual cloud predictions
    
    This validates that individual clouds add up to the integrated map
    """
    print("\n" + "="*60)
    print("COMPARING INTEGRATED MAP WITH INDIVIDUAL CLOUDS")
    print("="*60)
    
    # Load integrated map
    with fits.open(integrated_map_file) as hdul:
        integrated_data = hdul[0].data
        integrated_header = hdul[0].header
    
    print(f"\nIntegrated map shape: {integrated_data.shape}")
    print(f"Integrated total flux: {np.nansum(integrated_data):.2e}")
    
    # Load predictions
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
    
    print(f"Number of predicted clouds: {len(predictions)}")
    
    # Extract statistics
    mach_numbers = [p['mach_number'] for p in predictions]
    
    print(f"\nPredicted Mach numbers:")
    print(f"  Mean: {np.mean(mach_numbers):.2f} ± {np.std(mach_numbers):.2f}")
    print(f"  Range: [{np.min(mach_numbers):.2f}, {np.max(mach_numbers):.2f}]")
    
    # Check if clouds tile the integrated map
    print("\nNote: Integrated maps are survey-wide mosaics.")
    print("Individual clouds are extracted features, not guaranteed to tile perfectly.")
    print("Use integrated maps for:")
    print("  1. Understanding overall survey coverage")
    print("  2. Validating cloud extraction algorithm")
    print("  3. Checking coordinate systems")


def create_comparison_report(predictions_json, output_dir):
    """
    Create a comprehensive comparison report
    """
    output_dir = Path(output_dir)
    
    # Load predictions
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
    
    mach_numbers = np.array([p['mach_number'] for p in predictions])
    uncertainties = np.array([p['uncertainty'] for p in predictions])
    
    # Create detailed report
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main histogram
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(mach_numbers, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(np.mean(mach_numbers), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(mach_numbers):.2f}')
    ax1.axvline(np.median(mach_numbers), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(mach_numbers):.2f}')
    ax1.set_xlabel('Mach Number', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('MWIPS Survey: Mach Number Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Box plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.boxplot(mach_numbers, vert=True)
    ax2.set_ylabel('Mach Number')
    ax2.set_title('Box Plot')
    ax2.grid(alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_mach = np.sort(mach_numbers)
    ax3.plot(sorted_mach, np.linspace(0, 1, len(sorted_mach)), linewidth=2)
    ax3.set_xlabel('Mach Number')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('CDF')
    ax3.grid(alpha=0.3)
    
    # 4. Mach vs Uncertainty
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(mach_numbers, uncertainties, alpha=0.5, s=10)
    ax4.set_xlabel('Mach Number')
    ax4.set_ylabel('Uncertainty')
    ax4.set_title('Mach vs Uncertainty')
    ax4.grid(alpha=0.3)
    
    # 5. Percentile plot
    ax5 = fig.add_subplot(gs[1, 2])
    percentiles = [10, 25, 50, 75, 90]
    values = [np.percentile(mach_numbers, p) for p in percentiles]
    ax5.bar(range(len(percentiles)), values, tick_label=[f'{p}th' for p in percentiles])
    ax5.set_ylabel('Mach Number')
    ax5.set_title('Percentiles')
    ax5.grid(alpha=0.3)
    
    # 6. Statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    stats_table = [
        ['Statistic', 'Value'],
        ['Total Clouds', f'{len(mach_numbers)}'],
        ['Mean Mach', f'{np.mean(mach_numbers):.2f} ± {np.std(mach_numbers):.2f}'],
        ['Median Mach', f'{np.median(mach_numbers):.2f}'],
        ['Min Mach', f'{np.min(mach_numbers):.2f}'],
        ['Max Mach', f'{np.max(mach_numbers):.2f}'],
        ['10th percentile', f'{np.percentile(mach_numbers, 10):.2f}'],
        ['90th percentile', f'{np.percentile(mach_numbers, 90):.2f}'],
        ['Mean Uncertainty', f'{np.mean(uncertainties):.2f}']
    ]
    
    table = ax6.table(cellText=stats_table, cellLoc='center', loc='center',
                     bbox=[0.3, 0, 0.4, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    fig.suptitle('MWIPS Survey: Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    plt.savefig(output_dir / 'mwips_comprehensive_report.png', dpi=150, bbox_inches='tight')
    print(f"\nComprehensive report saved: {output_dir / 'mwips_comprehensive_report.png'}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    INTEGRATED_MAPS_DIR = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/CO_L105_150_B-5_5_V-95_25_integrated_maps/'
    PREDICTIONS_JSON = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/predictions/mwips_predictions.json'
    CLOUDS_DIR = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/clouds_q2_CO/clouds_13co/clouds_13co_fits/'
    OUTPUT_DIR = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/predictions/'
    
    print("MWIPS INTEGRATED MAPS INSPECTOR")
    
    # Step 1: Inspect integrated maps
    if Path(INTEGRATED_MAPS_DIR).exists():
        inspect_integrated_maps(INTEGRATED_MAPS_DIR)
    else:
        print(f"Integrated maps directory not found: {INTEGRATED_MAPS_DIR}")
    
    # Step 2: Compare with predictions (if available)
    if Path(PREDICTIONS_JSON).exists():
        print("\n\nCreating comparison report...")
        create_comparison_report(PREDICTIONS_JSON, OUTPUT_DIR)
    else:
        print(f"\nPredictions file not found: {PREDICTIONS_JSON}")
        print("Run process_MWIPS_survey.py first to generate predictions")