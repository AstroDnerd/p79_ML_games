"""
MWIPS Survey Batch Processing Script
Process ~3000 molecular cloud 13CO datacubes and predict Mach numbers
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import torch
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your existing functions
sys.path.append('.')  # Adjust path as needed
from get_Observation_Data import (
    load_ppv_cube, 
    compute_moment_maps, 
    prepare_model_input
)
from do_Observation_ViT_test import load_trained_model


class MWIPSProcessor:
    """Process MWIPS molecular cloud survey data"""
    
    def __init__(self, data_dir, output_dir, model_path='./models/test9005.pth'):
        """
        Initialize processor
        
        Args:
            data_dir: Path to clouds_q2_CO/clouds_13co/clouds_13co_fits/
            output_dir: Where to save results
            model_path: Path to trained model
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output subdirectories
        self.img_dir = self.output_dir / 'individual_predictions'
        self.img_dir.mkdir(exist_ok=True)
        
        # Load model once
        print("Loading model...")
        self.model, self.has_uncertainty = load_trained_model(model_path, 'net9005')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Storage for results
        self.results = []
        self.statistics = {}
        
    def scan_cloud_statistics(self, min_spatial_size=128):
        """
        First pass: scan all clouds and collect statistics
        
        Args:
            min_spatial_size: Minimum spatial dimension required (pixels)
        
        Returns:
            valid_clouds: List of cloud files meeting criteria
            stats_dict: Dictionary of statistics for all clouds
        """
        print("STEP 1: SCANNING CLOUD STATISTICS")
        
        cloud_files = sorted(self.data_dir.glob('cloud*cube_13CO_dbscan_2sigma.fits'))
        print(f"Found {len(cloud_files)} cloud files")
        
        stats_list = []
        valid_clouds = []
        valid_after_upsample = []
        
        for fits_file in tqdm(cloud_files, desc="Scanning clouds"):
            try:
                with fits.open(fits_file) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    # Extract cloud ID from filename
                    cloud_id = fits_file.stem.split('cloud')[1].split('cube')[0]
                    
                    # Get dimensions
                    if data.ndim == 3:
                        n_vel, n_y, n_x = data.shape
                    else:
                        continue  # Skip if not 3D
                    
                    # File size
                    file_size_mb = fits_file.stat().st_size / (1024 * 1024)
                    
                    # Check for emission
                    n_nonzero = np.sum(~np.isnan(data) & (data != 0))
                    emission_fraction = n_nonzero / data.size
                    
                    # Spatial coverage
                    spatial_size = min(n_y, n_x)
                    
                    stats = {
                        'cloud_id': cloud_id,
                        'filename': fits_file.name,
                        'file_size_mb': file_size_mb,
                        'n_vel': n_vel,
                        'n_y': n_y,
                        'n_x': n_x,
                        'upsample_factor': max(1, np.ceil(min_spatial_size / spatial_size)),
                        'spatial_size': spatial_size,
                        'emission_fraction': emission_fraction,
                        'valid': spatial_size >= min_spatial_size and emission_fraction > 0.01,
                        'valid_after_upsample': (spatial_size * max(1, np.ceil(min_spatial_size / spatial_size))) >= min_spatial_size and emission_fraction > 0.01
                    }
                    
                    stats_list.append(stats)
                    
                    if stats['valid']:
                        valid_clouds.append(fits_file)
                    if stats['valid_after_upsample']:
                        valid_after_upsample.append(fits_file)
                        
            except Exception as e:
                print(f"Error scanning {fits_file.name}: {e}")
                continue
        
        # Create statistics summary
        stats_array = np.array([
            [s['file_size_mb'], s['spatial_size'], s['emission_fraction']] 
            for s in stats_list
        ])
        
        self.statistics = {
            'total_clouds': len(cloud_files),
            'valid_clouds': len(valid_clouds),
            'valid_after_upsample': len(valid_after_upsample),
            'invalid_clouds': len(cloud_files) - len(valid_clouds),
            'invalid_after_upsample': len(cloud_files) - len(valid_after_upsample),
            'file_size': {
                'min': stats_array[:, 0].min(),
                'max': stats_array[:, 0].max(),
                'mean': stats_array[:, 0].mean(),
                'median': np.median(stats_array[:, 0])
            },
            'spatial_size': {
                'min': int(stats_array[:, 1].min()),
                'max': int(stats_array[:, 1].max()),
                'mean': stats_array[:, 1].mean(),
                'median': np.median(stats_array[:, 1])
            },
            'emission_fraction': {
                'min': stats_array[:, 2].min(),
                'max': stats_array[:, 2].max(),
                'mean': stats_array[:, 2].mean(),
                'median': np.median(stats_array[:, 2])
            },
            'all_stats': stats_list
        }
        
        # Print summary
        print("STATISTICS SUMMARY")
        print(f"Total clouds: {self.statistics['total_clouds']}")
        print(f"Valid clouds (>={min_spatial_size}px): {self.statistics['valid_clouds']}")
        print(f"Valid after upsample: {self.statistics['valid_after_upsample']}")
        print(f"Invalid clouds: {self.statistics['invalid_clouds']}")
        print(f"Invalid after upsample: {self.statistics['invalid_after_upsample']}")
        print(f"\nFile size (MB): {self.statistics['file_size']['min']:.2f} - {self.statistics['file_size']['max']:.2f} (median: {self.statistics['file_size']['median']:.2f})")
        print(f"Spatial size (px): {self.statistics['spatial_size']['min']} - {self.statistics['spatial_size']['max']} (median: {self.statistics['spatial_size']['median']:.0f})")
        print(f"Emission fraction: {self.statistics['emission_fraction']['min']:.3f} - {self.statistics['emission_fraction']['max']:.3f} (median: {self.statistics['emission_fraction']['median']:.3f})")
        
        # Save statistics
        with open(self.output_dir / 'cloud_statistics.json', 'w') as f:
            # Convert numpy types to Python types for JSON
            stats_json = {
                'total_clouds': int(self.statistics['total_clouds']),
                'valid_clouds': int(self.statistics['valid_clouds']),
                'valid_after_upsample': int(self.statistics['valid_after_upsample']),
                'invalid_clouds': int(self.statistics['invalid_clouds']),
                'invalid_after_upsample': int(self.statistics['invalid_after_upsample']),
                'file_size': {k: float(v) for k, v in self.statistics['file_size'].items()},
                'spatial_size': {k: float(v) for k, v in self.statistics['spatial_size'].items()},
                'emission_fraction': {k: float(v) for k, v in self.statistics['emission_fraction'].items()}
            }
            json.dump(stats_json, f, indent=2)
        
        # Create statistics visualization
        self.visualize_statistics(stats_list)
        
        return valid_clouds, valid_after_upsample, self.statistics
    
    def visualize_statistics(self, stats_list):
        """Create visualization of cloud statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        file_sizes = [s['file_size_mb'] for s in stats_list]
        spatial_sizes = [s['spatial_size'] for s in stats_list]
        emission_fracs = [s['emission_fraction'] for s in stats_list]
        valid = [s['valid'] for s in stats_list]
        valid_after_upsample = [s['valid_after_upsample'] for s in stats_list]
        
        # File size distribution
        axes[0, 0].hist(file_sizes, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('File Size (MB)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('File Size Distribution')
        axes[0, 0].axvline(np.median(file_sizes), color='red', linestyle='--', label='Median')
        axes[0, 0].legend()
        
        # Spatial size distribution
        axes[0, 1].hist(spatial_sizes, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_xlabel('Spatial Size (pixels)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Spatial Size Distribution')
        axes[0, 1].axvline(128, color='red', linestyle='--', label='Min threshold (128px)')
        axes[0, 1].legend()
        
        # Emission fraction distribution
        axes[1, 0].hist(emission_fracs, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_xlabel('Emission Fraction')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Emission Fraction Distribution')
        axes[1, 0].set_yscale('log')
        
        # Valid vs Invalid
        valid_count = sum(valid)
        invalid_count = len(valid) - valid_count
        invalid_after_upsample_count = sum([not v for v in valid_after_upsample])
        axes[1, 1].bar(['Valid', 'Invalid', 'Invalid After Upsample'], 
                      [valid_count, invalid_count, invalid_after_upsample_count], 
                      color=['green', 'red', 'orange'], alpha=0.7)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Cloud Validation Summary')
        axes[1, 1].text(0, valid_count + 50, f'{valid_count}', ha='center', fontweight='bold')
        axes[1, 1].text(1, invalid_count + 50, f'{invalid_count}', ha='center', fontweight='bold')
        axes[1, 1].text(2, invalid_after_upsample_count + 50, f'{invalid_after_upsample_count}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cloud_statistics.png', dpi=150)
        print(f"\nStatistics saved: {self.output_dir / 'cloud_statistics.png'}")
        plt.close()
    
    def process_single_cloud(self, fits_file):
        """
        Process a single cloud: load -> moment maps -> predict
        Does NOT save intermediate files (memory efficient)
        
        Args:
            fits_file: Path to FITS file
        
        Returns:
            result_dict: Dictionary with predictions and metadata
        """
        try:
            # Extract cloud ID
            cloud_id = fits_file.stem.split('cloud')[1].split('cube')[0]
            
            # Step 1: Load PPV cube
            data, header, wcs, velocity_axis = load_ppv_cube(str(fits_file), data_format='auto')
            
            # Step 2: Compute moment maps
            moment0, moment1, moment2, emission_mask = compute_moment_maps(data, velocity_axis, noise_threshold=2.0)
            
            ny, nx = moment0.shape
            if min(ny, nx) < 128:
                from get_Observation_Data import upsample_moment_maps
                moment0, moment1, moment2, emission_mask = upsample_moment_maps(
                    moment0, moment1, moment2, emission_mask, target_size=128
                )
            # Check if sufficient emission
            if emission_mask.sum() < 100:
                return None  # Skip clouds with insufficient emission
            
            # Step 3: Prepare model input (single 128x128)
            model_input = prepare_model_input(moment0, moment1, moment2, emission_mask, target_size=128)
            
            # Step 4: Run inference
            input_tensor = torch.from_numpy(model_input).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            if self.has_uncertainty:
                mean, logvar = output
                mach_pred = mean[0][0].cpu().item()
                mach_unc = torch.exp(0.5 * logvar[0][0]).cpu().item()
            else:
                mach_pred = output[0][0].cpu().item()
                mach_unc = 0.0
            
            # Step 5: Create visualization (save this)
            self.save_cloud_prediction(cloud_id, moment0, emission_mask, mach_pred, mach_unc)
            
            # Step 6: Collect results
            result = {
                'cloud_id': cloud_id,
                'filename': fits_file.name,
                'mach_number': mach_pred,
                'uncertainty': mach_unc,
                'spatial_shape': moment0.shape,
                'emission_pixels': int(emission_mask.sum()),
                'emission_fraction': emission_mask.sum() / emission_mask.size,
                'moment0_max': float(np.nanmax(moment0)),
                'moment2_mean': float(np.nanmean(moment2[emission_mask]))
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {fits_file.name}: {e}")
            return None
    
    def save_cloud_prediction(self, cloud_id, moment0, emission_mask, mach_pred, mach_unc):
        """Save individual cloud prediction as image"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Show moment0 map
        im = ax.imshow(moment0, origin='lower', cmap='viridis')
        ax.contour(emission_mask, levels=[0.5], colors='white', linewidths=1, alpha=0.5)
        
        plt.colorbar(im, ax=ax, label='Integrated Intensity [K km/s]')
        
        # Title with prediction
        if mach_unc > 0:
            title = f'Cloud {cloud_id}: M = {mach_pred:.2f} ± {mach_unc:.2f}'
        else:
            title = f'Cloud {cloud_id}: M = {mach_pred:.2f}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        
        plt.tight_layout()
        plt.savefig(self.img_dir / f'cloud_{cloud_id}_prediction.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def process_all_clouds(self, valid_clouds, max_clouds=None):
        """
        Process all valid clouds
        
        Args:
            valid_clouds: List of cloud files to process
            max_clouds: Limit processing (for testing, None = all)
        """
        print("STEP 2: PROCESSING CLOUDS AND PREDICTING")
        
        if max_clouds:
            valid_clouds = valid_clouds[:max_clouds]
            print(f"Processing first {max_clouds} clouds (test mode)")
        
        for fits_file in tqdm(valid_clouds, desc="Processing clouds"):
            result = self.process_single_cloud(fits_file)
            
            if result:
                self.results.append(result)
        
        print(f"\nSuccessfully processed: {len(self.results)} clouds")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save all results to JSON and create summary plots"""
        # Save JSON
        results_file = self.output_dir / 'mwips_predictions.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved: {results_file}")
        
        # Create summary histogram
        self.create_summary_histogram()
    
    def create_summary_histogram(self):
        """Create final summary histogram of all predictions"""
        if not self.results:
            print("No results to plot")
            return
        
        mach_numbers = [r['mach_number'] for r in self.results]
        uncertainties = [r['uncertainty'] for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Main histogram
        axes[0, 0].hist(mach_numbers, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].axvline(np.mean(mach_numbers), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(mach_numbers):.2f}')
        axes[0, 0].axvline(np.median(mach_numbers), color='orange', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(mach_numbers):.2f}')
        axes[0, 0].set_xlabel('Mach Number', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title(f'MWIPS Mach Number Distribution (N={len(mach_numbers)})', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Cumulative distribution
        axes[0, 1].hist(mach_numbers, bins=50, cumulative=True, density=True,
                       alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_xlabel('Mach Number', fontsize=12)
        axes[0, 1].set_ylabel('Cumulative Probability', fontsize=12)
        axes[0, 1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Uncertainty histogram
        axes[1, 0].hist(uncertainties, bins=30, alpha=0.7, edgecolor='black', color='coral')
        axes[1, 0].axvline(np.mean(uncertainties), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(uncertainties):.2f}')
        axes[1, 0].set_xlabel('Uncertainty', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Prediction Uncertainty Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Statistics table
        axes[1, 1].axis('off')
        stats_text = f"""
MWIPS SURVEY STATISTICS

Total Clouds Processed: {len(mach_numbers)}

Mach Number Statistics:
  Mean:   {np.mean(mach_numbers):>6.2f} ± {np.std(mach_numbers):.2f}
  Median: {np.median(mach_numbers):>6.2f}
  Range:  [{np.min(mach_numbers):.2f}, {np.max(mach_numbers):.2f}]
  
Percentiles:
  25th:   {np.percentile(mach_numbers, 25):>6.2f}
  50th:   {np.percentile(mach_numbers, 50):>6.2f}
  75th:   {np.percentile(mach_numbers, 75):>6.2f}

Uncertainty:
  Mean:   {np.mean(uncertainties):>6.2f}
  Range:  [{np.min(uncertainties):.2f}, {np.max(uncertainties):.2f}]
"""
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mwips_summary.png', dpi=150, bbox_inches='tight')
        print(f"Summary saved: {self.output_dir / 'mwips_summary.png'}")
        plt.close()
        
        # Print to console
        print("FINAL STATISTICS")
        print(f"Processed clouds: {len(mach_numbers)}")
        print(f"Mean Mach: {np.mean(mach_numbers):.2f} ± {np.std(mach_numbers):.2f}")
        print(f"Median Mach: {np.median(mach_numbers):.2f}")
        print(f"Range: [{np.min(mach_numbers):.2f}, {np.max(mach_numbers):.2f}]")


def check_integrated_maps(integrated_dir):
    """
    Check what the integrated map files contain
    
    Args:
        integrated_dir: Path to CO_L105_150_B-5_5_V-95_25_integrated_maps/
    """
    print("CHECKING INTEGRATED MAPS")
    
    integrated_dir = Path(integrated_dir)
    
    for fits_file in integrated_dir.glob('*.fits'):
        print(f"\nFile: {fits_file.name}")
        print(f"Size: {fits_file.stat().st_size / (1024**2):.2f} MB")
        
        with fits.open(fits_file) as hdul:
            hdul.info()
            print(f"\nHeader info:")
            print(f"  NAXIS: {hdul[0].header.get('NAXIS')}")
            if hdul[0].header.get('NAXIS') >= 2:
                print(f"  Shape: {hdul[0].data.shape}")
                print(f"  CTYPE1: {hdul[0].header.get('CTYPE1')}")
                print(f"  CTYPE2: {hdul[0].header.get('CTYPE2')}")
                if hdul[0].header.get('NAXIS') == 3:
                    print(f"  CTYPE3: {hdul[0].header.get('CTYPE3')}")
            
            # Check data range
            data = hdul[0].data
            print(f"  Data range: {np.nanmin(data):.3f} to {np.nanmax(data):.3f}")
            print(f"  Non-zero fraction: {np.sum(data != 0) / data.size:.3%}")


def main():
    """Main execution"""
    # Configuration
    DATA_DIR = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/clouds_q2_CO/clouds_13co/clouds_13co_fits/'
    OUTPUT_DIR = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/predictions/'
    INTEGRATED_MAPS_DIR = '/home/x-nbisht1/projects/p79d_dataset/observations/mwips/CO_L105_150_B-5_5_V-95_25_integrated_maps/'
    MODEL_PATH = '/home/x-nbisht1/projects/p79d_dataset/models/test9005.pth'
    
    # Optional: Check integrated maps first
    if os.path.exists(INTEGRATED_MAPS_DIR):
        check_integrated_maps(INTEGRATED_MAPS_DIR)
    
    # Initialize processor
    processor = MWIPSProcessor(DATA_DIR, OUTPUT_DIR, MODEL_PATH)
    
    # Step 1: Scan statistics
    valid_clouds, valid_after_upsample, stats = processor.scan_cloud_statistics(min_spatial_size=128)
    
    print(f"\nRecommendation: Process {len(valid_clouds)} valid clouds")
    print(f"Valid after upsample: {len(valid_after_upsample)}")
    
    # Ask user to confirm
    response = input("\nProceed with processing? (y/n, or enter max number to test): ")
    
    if response.lower() == 'n':
        print("Exiting without processing")
        return
    elif response.lower() == 'y':
        max_clouds = None
    else:
        try:
            max_clouds = int(response)
        except:
            max_clouds = 10
    
    # Ask user to confirm
    response = input("\nValid cloud or Valid after upsampling? (v/u): ")
    clouds = valid_clouds
    if response.lower() == 'u':
        clouds = valid_after_upsample
    
    # Step 2: Process all valid clouds
    processor.process_all_clouds(clouds, max_clouds=max_clouds)
    
    print("MWIPS PROCESSING COMPLETE")
    print(f"Results directory: {OUTPUT_DIR}")
    print(f"  - cloud_statistics.json: Cloud metadata")
    print(f"  - cloud_statistics.png: Statistics visualization")
    print(f"  - mwips_predictions.json: All predictions")
    print(f"  - mwips_summary.png: Summary histogram")
    print(f"  - individual_predictions/: Individual cloud images ({len(processor.results)} files)")


if __name__ == "__main__":
    main()