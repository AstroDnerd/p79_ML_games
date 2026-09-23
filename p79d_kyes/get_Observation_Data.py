import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import zoom
import requests
import os
from pathlib import Path

#TAURUS DATA: https://lweb.cfa.harvard.edu/rtdc/CO/NumberedRegions/DHT21/index.html
#PERSEUS DATA: https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:10904/10075&studyListingIndex=9_146394a15e2efcc6f22d801ac7ff
#SERPENS DATA: http://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A%2BA/646/A170
#ORION DATA: https://cdsarc.cds.unistra.fr/ftp/J/A+A/564/A68/fits/
#Ophiocus: https://lweb.cfa.harvard.edu/COMPLETE/data/mol_lines/FCRAO/

def load_ppv_cube(fits_path, data_format='auto'):
    """
    Load PPV cube from FITS file and extract relevant information.
    Now handles both Perseus (xyv format) and Taurus (vlb format) data.
    
    Args:
        fits_path: Path to FITS file
        data_format: 'auto', 'perseus', or 'taurus'
    
    Returns:
        data: 3D array (velocity, y, x) in K (brightness temperature)
        header: FITS header with WCS information
        wcs: World Coordinate System object
        velocity_axis: Velocity values in km/s
    """
    print(f"\nLoading PPV cube from: {fits_path}")
    
    with fits.open(fits_path) as hdul:
        data_raw = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    
    print(f"Raw data shape: {data_raw.shape}")
    print(f"Data units: {header.get('BUNIT', 'Unknown')}")
    
    # Auto-detect format based on CTYPE keywords
    if data_format == 'auto':
        ctype1 = header.get('CTYPE1', '').upper()
        ctype2 = header.get('CTYPE2', '').upper()
        ctype3 = header.get('CTYPE3', '').upper()
        
        print(f"Axis types: CTYPE1={ctype1}, CTYPE2={ctype2}, CTYPE3={ctype3}")
        if 'VELO' in ctype1 or 'VRAD' in ctype1:
            data_format = 'taurus'  #Taurus: (velo, glon, glat)
            print("Detected Taurus format (vlb)")
        elif 'VELO' in ctype3 or 'VRAD' in ctype3:
            data_format = 'perseus'  #Perseus: (ra, dec, velo)
            print("Detected Perseus format (xyv)")
        else:
            raise ValueError(f"Cannot auto-detect format. CTYPE1={ctype1}, CTYPE3={ctype3}")
    
    #Handle different formats
    if data_format == 'taurus':
        #Taurus format: FITS axes are (VELO, GLON, GLAT)
        # FITS loads as [NAXIS3, NAXIS2, NAXIS1] = [GLAT, GLON, VELO]
        # We need (VELO, GLAT, GLON) or (VELO, Y, X)
        
        print("Processing (VELO, GLON, GLAT) format...")
        
        #FITS loads in reverse order: data_raw is (lat, lon, velo)
        #Transpose to (velo, lat, lon)
        data = np.transpose(data_raw, (2, 0, 1))
        print(f"Transposed to shape: {data.shape} (velocity, lat, lon)")
        
        # Get velocity axis parameters from AXIS1 (velocity)
        naxis1 = header['NAXIS1']  # Number of velocity channels
        crval1 = header['CRVAL1']  # Reference velocity in km/s
        cdelt1 = header['CDELT1']  # Velocity channel width in km/s
        crpix1 = header['CRPIX1']  # Reference pixel
        
        # Velocity already in km/s
        velocity_axis = crval1 + (np.arange(naxis1) - (crpix1 - 1)) * cdelt1
        
        print(f"Velocity range: {velocity_axis.min():.2f} to {velocity_axis.max():.2f} km/s")
        print(f"Velocity resolution: {cdelt1:.3f} km/s")
        
    elif data_format == 'perseus':
        #Perseus format: FITS axes are (RA, DEC, VELO)
        #FITS loads as [NAXIS3, NAXIS2, NAXIS1] = [VELO, DEC, RA]
        #Already in correct order (velo, y, x)
        
        print("Processing Perseus format...")
        data = data_raw
        
        # Get velocity axis parameters from AXIS3 (velocity)
        naxis3 = header['NAXIS3']
        crval3 = header['CRVAL3']  # Reference velocity in m/s
        cdelt3 = header['CDELT3']  # Velocity channel width in m/s
        crpix3 = header['CRPIX3']  # Reference pixel
        
        # Convert velocity from m/s to km/s
        velocity_axis = (crval3 + (np.arange(naxis3) - (crpix3 - 1)) * cdelt3) / 1000.0
        
        print(f"Velocity range: {velocity_axis.min():.2f} to {velocity_axis.max():.2f} km/s")
        print(f"Velocity resolution: {cdelt3/1000.0:.3f} km/s")
    
    else:
        raise ValueError(f"Unknown data format: {data_format}")
    
    print(f"Final data shape: {data.shape} (velocity, y, x)")
    print(f"Spatial shape: {data.shape[1]} x {data.shape[2]} pixels")
    
    return data, header, wcs, velocity_axis


def compute_moment_maps(data, velocity_axis, noise_threshold=3.0):
    """
    Compute the three moment maps from PPV cube.
    
    Moment 0: Integrated intensity ∫ T_B dv (proxy for column density)
    Moment 1: Velocity centroid ∫ v·T_B dv / ∫ T_B dv
    Moment 2: Velocity dispersion sqrt(∫ (v-v̄)²·T_B dv / ∫ T_B dv)
    
    Args:
        data: PPV cube [velocity, y, x] in K
        velocity_axis: Velocity values in km/s
        noise_threshold: Sigma threshold for masking noise
    
    Returns:
        moment0: Integrated intensity [K km/s]
        moment1: Velocity centroid [km/s]
        moment2: Velocity dispersion [km/s]
    """
    print("\nComputing moment maps...")
    
    # Handle BLANK values (marked as -32768 in Taurus data)
    data_clean = np.copy(data)
    data_clean[data_clean < -1000] = np.nan

    #Estimate noise from emission-free regions, use edge channels assumed to have no line emission
    edge_channels = 5
    noise_estimate = np.nanstd(np.concatenate([
        data_clean[:edge_channels].flatten(),
        data_clean[-edge_channels:].flatten()
    ]))
    
    print(f"Estimated noise: {noise_estimate:.4f} K")
    threshold = noise_threshold * noise_estimate
    print(f"Applying {noise_threshold}σ threshold: {threshold:.4f} K")
    
    #Create mask where emission is significant
    mask = data_clean > threshold
    
    #Velocity channel width (assuming uniform spacing)
    dv = np.abs(velocity_axis[1] - velocity_axis[0])
    
    # Moment 0: Integrated intensity, for each (y, x) pixel, sum T_B across velocity weighted by channel width
    moment0 = np.nansum(np.where(mask, data_clean, 0.0), axis=0) * dv
    
    emission_mask = moment0 > (noise_threshold * noise_estimate * dv * np.sqrt(len(velocity_axis)))
    
    #Moment 1: Velocity centroid, weighted average: v̄ = Σ(v·T_B·dv) / Σ(T_B·dv)
    velocity_3d = velocity_axis[:, np.newaxis, np.newaxis]
    numerator = np.nansum(np.where(mask, velocity_3d * data_clean, 0.0), axis=0) * dv
    moment1 = np.where(emission_mask, numerator / moment0, np.nan)
    
    #Moment 2: Velocity dispersion, σ_v = sqrt(Σ((v-v̄)²·T_B·dv) / Σ(T_B·dv)) and expand moment1 to 3D for broadcasting
    moment1_3d = moment1[np.newaxis, :, :]
    velocity_deviation_sq = (velocity_3d - moment1_3d) ** 2
    variance = np.nansum(np.where(mask, velocity_deviation_sq * data_clean, 0.0), axis=0) * dv
    moment2 = np.where(emission_mask, np.sqrt(variance / moment0), np.nan)
    
    print(f"Moment 0 range: {np.nanmin(moment0):.2f} to {np.nanmax(moment0):.2f} K km/s")
    print(f"Moment 1 range: {np.nanmin(moment1):.2f} to {np.nanmax(moment1):.2f} km/s")
    print(f"Moment 2 range: {np.nanmin(moment2):.2f} to {np.nanmax(moment2):.2f} km/s")
    print(f"Pixels with emission: {np.sum(emission_mask)} / {emission_mask.size}")
    
    return moment0, moment1, moment2, emission_mask

def upsample_moment_maps(moment0, moment1, moment2, emission_mask, target_size=128):
    """
    Upsample moment maps to meet minimum spatial size requirement.
    
    For small clouds (<128x128), this increases spatial sampling by interpolation.
    Uses bilinear interpolation to preserve smoothness.
    Also check for clouds too small
    
    Args:
        moment0, moment1, moment2: Moment maps
        emission_mask: Boolean emission mask
        target_size: Minimum size required (default 128)
    
    Returns:
        Upsampled moment0, moment1, moment2, emission_mask
    """
    ny, nx = moment0.shape
    current_size = min(ny, nx)
    
    # Check if upsampling needed
    if current_size >= target_size:
        print(f"Spatial size {current_size} >= {target_size}, no upsampling needed")
        return moment0, moment1, moment2, emission_mask
    
    # Calculate upsample factor to reach target_size
    upsample_factor = np.ceil(target_size / current_size)
    if upsample_factor > 5.0:
        print(f"Spatial size {current_size} too small, upsampling factor {upsample_factor} > 5.0, not upsampling.")
        return moment0, moment1, moment2, emission_mask
    
    print(f"\nUpsampling moment maps:")
    print(f"  Original size: {ny} x {nx} (min: {current_size})")
    print(f"  Upsample factor: {upsample_factor}x")
    print(f"  Target size: {int(ny * upsample_factor)} x {int(nx * upsample_factor)}")
    
    # Upsample each moment map using bilinear interpolation (order=1)
    # order=1 is bilinear, order=3 is bicubic
    # Bilinear is smoother and better for physical fields
    
    moment0_upsampled = zoom(moment0, upsample_factor, order=1)
    moment1_upsampled = zoom(moment1, upsample_factor, order=1)
    moment2_upsampled = zoom(moment2, upsample_factor, order=1)
    
    # For emission mask, use nearest neighbor (order=0) to preserve boolean nature
    emission_mask_upsampled = zoom(emission_mask.astype(float), upsample_factor, order=0).astype(bool)
    
    # Verify sizes
    ny_new, nx_new = moment0_upsampled.shape
    print(f"  Final size: {ny_new} x {nx_new}")
    
    # Sanity check: make sure NaN structure is preserved
    # In upsampled regions, NaNs might be interpolated - restore them
    # based on emission mask
    moment1_upsampled = np.where(emission_mask_upsampled, moment1_upsampled, np.nan)
    moment2_upsampled = np.where(emission_mask_upsampled, moment2_upsampled, np.nan)
    
    return moment0_upsampled, moment1_upsampled, moment2_upsampled, emission_mask_upsampled

def convert_to_column_density(moment0, transition='13CO'):
    """
    Convert integrated intensity to column density.
    
    For optically thin 13CO(1-0):
    N(13CO) = 2.4 x 10^14 x ∫ T_B dv [cm^-2] (for T_ex = 10 K)
    
    For 12CO (optically thick), use empirical relations:
    N(H2) ≈ X_CO x ∫ T_B(12CO) dv
    where X_CO ≈ 2 x 10^20 cm^-2 (K km/s)^-1 (Galactic value)
    
    Args:
        moment0: Integrated intensity [K km/s]
        transition: '12CO' or '13CO'
    
    Returns:
        column_density: N(H2) [cm^-2]
    """
    print(f"\nConverting {transition} integrated intensity to H2 column density...")
    
    if transition == '13CO':
        # 13CO to H2 conversion
        # Assume [13CO]/[H2] ≈ 1.5 × 10^-6 (local ISM value)
        # And optically thin: N(13CO) = C × ∫ T_B dv
        
        # Excitation temperature assumption
        T_ex = 10.0  # K, typical for molecular clouds
        
        # Conversion factor (see Pineda et al. 2008, ApJ, 679, 481)
        # N(13CO) [cm^-2] = 2.4 × 10^14 × (T_ex / (T_ex - 2.7)) × ∫ T_B dv
        factor = 2.4e14 * (T_ex / (T_ex - 2.7))
        N_13CO = factor * moment0  # cm^-2
        
        # Convert to H2 using abundance ratio
        abundance_ratio = 1.5e-6  # [13CO]/[H2]
        column_density = N_13CO / abundance_ratio
        
    elif transition == '12CO':
        # Standard X-factor conversion (optically thick)
        X_CO = 2.0e20  # cm^-2 (K km/s)^-1, Galactic value
        column_density = X_CO * moment0
    
    else:
        raise ValueError(f"Unknown transition: {transition}")
    
    print(f"Column density range: {np.nanmin(column_density):.2e} to {np.nanmax(column_density):.2e} cm^-2")
    
    #Convert to surface density (assuming standard cloud depth ~ 1 pc = 3.086e18 cm)
    #This gives approximate volume density for comparison with simulations
    cloud_depth = 3.086e18  # cm (1 pc)
    volume_density = column_density / cloud_depth  # cm^-3
    print(f"Implied volume density: {np.nanmin(volume_density):.2e} to {np.nanmax(volume_density):.2e} cm^-3")
    
    return column_density

def prepare_model_input(moment0, moment1, moment2, emission_mask, target_size=128):
    """
    Prepare 3-channel input matching simulation training data format.
    
    Simulation channels:
    - Channel 0: Column density (density integrated along LoS)
    - Channel 1: Velocity-weighted column density
    - Channel 2: Velocity dispersion
    
    Args:
        moment0: Integrated intensity [proxy for column density]
        moment1: Velocity centroid [km/s]
        moment2: Velocity dispersion [km/s]
        emission_mask: Boolean mask of emission regions
        target_size: Output spatial size (default: 128)
    
    Returns:
        model_input: Array of shape (3, target_size, target_size)
    """
    print(f"\nPreparing model input (target size: {target_size}x{target_size})...")
    
    # Replace NaNs with zeros (no emission regions)
    moment0_clean = np.nan_to_num(moment0, nan=0.0)
    moment1_clean = np.nan_to_num(moment1, nan=0.0)
    moment2_clean = np.nan_to_num(moment2, nan=0.0)
    
    # Channel 0: Column density (use moment0 as proxy)
    # Normalize to reasonable range similar to simulations
    channel0 = moment0_clean
    
    # Channel 1: Velocity-weighted column density
    # In observations: ∫ v·T_B dv (already computed in moment1 numerator)
    # Reconstruct: (moment1 * moment0) gives velocity-weighted integral
    channel1 = moment1_clean * moment0_clean
    
    # Channel 2: Velocity dispersion
    # Note: In simulations, this might be sqrt(σ_v²), but observations give σ_v directly
    # Some simulations use σ_v², so check if squaring improves results
    channel2 = moment2_clean**2  # Try both moment2 and moment2**2
    
    # Crop or pad to square
    ny, nx = channel0.shape
    size = min(ny, nx)
    res_factor = min(1.0, size / target_size)

    # Center crop
    y_start = (ny - size) // 2
    x_start = (nx - size) // 2
    
    channel0 = channel0[y_start:y_start+size, x_start:x_start+size]
    channel1 = channel1[y_start:y_start+size, x_start:x_start+size]
    channel2 = channel2[y_start:y_start+size, x_start:x_start+size]
    
    # Resize to target size
    zoom_factor = target_size / size
    
    channel0 = zoom(channel0, zoom_factor, order=1)  # Bilinear interpolation
    channel1 = zoom(channel1, zoom_factor, order=1)
    channel2 = zoom(channel2, zoom_factor, order=1)
    
    # Stack into (3, H, W) format
    model_input = np.stack([channel0, channel1, channel2], axis=0)
    
    # Normalize each channel (important for model performance)
    # Option 1: Standardize (zero mean, unit variance)
    for i in range(3):
        channel = model_input[i]
        if np.std(channel) > 0:
            model_input[i] = (channel - np.mean(channel)) / np.std(channel)
    
    print(f"Output shape: {model_input.shape}")
    print(f"Channel 0 (density) range: {model_input[0].min():.3f} to {model_input[0].max():.3f}")
    print(f"Channel 1 (velocity-weighted) range: {model_input[1].min():.3f} to {model_input[1].max():.3f}")
    print(f"Channel 2 (dispersion) range: {model_input[2].min():.3f} to {model_input[2].max():.3f}")
    
    return model_input, res_factor

def prepare_tiled_model_input(moment0, moment1, moment2, emission_mask, 
                               n_tiles=10, min_coverage=0.10, max_overlap=0.1, was_upsampled=False, original_min_size=128):
    """
    Intelligently sample n_tiles from emission regions using density-based selection.
    
    Strategy:
    1. Find ALL possible 128x128 positions
    2. Score each by emission density
    3. Greedily select top n_tiles with controlled overlap
    
    Args:
        moment0, moment1, moment2: Moment maps
        emission_mask: Boolean mask
        n_tiles: EXACT number of tiles to extract
        min_coverage: Minimum fraction of tile with emission (0-1)
        max_overlap: Maximum allowed overlap between tiles (0-1)
    
    Returns:
        Dict with tiles, positions, and metadata
    """
    print(f"\nPreparing {n_tiles} tiles using emission-based sampling...")
    
    ny, nx = moment0.shape
    tile_size = 128
    tile_res_factor = min(1.0, original_min_size / 128) if was_upsampled else 1.0

    # Step 1: Generate ALL possible tile positions and score them
    print("Scanning for emission-rich regions...")
    candidates = []
    
    # Use smaller stride for finer sampling
    stride = 32  # Small stride to catch all good positions
    
    for y in range(0, ny - tile_size + 1, stride):
        for x in range(0, nx - tile_size + 1, stride):
            mask_tile = emission_mask[y:y+tile_size, x:x+tile_size]
            coverage = mask_tile.sum() / (tile_size * tile_size)
            
            if coverage >= min_coverage:
                # Score by total emission (higher = better)
                m0_tile = moment0[y:y+tile_size, x:x+tile_size]
                emission_score = np.nansum(m0_tile * mask_tile)
                
                candidates.append({
                    'position': (y, x),
                    'coverage': coverage,
                    'score': emission_score
                })
    
    print(f"Found {len(candidates)} candidate positions")
    
    if len(candidates) == 0:
        raise ValueError(f"No valid tiles found with coverage >= {min_coverage:.1%}")
    
    # Step 2: Greedily select n_tiles with overlap control
    print("Selecting best non-overlapping tiles...")
    
    # Sort by score (best first)
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    selected = []
    selected_boxes = []  # Track bounding boxes to compute overlap
    
    for candidate in candidates:
        if len(selected) >= n_tiles:
            break
        
        y, x = candidate['position']
        new_box = (y, x, y + tile_size, x + tile_size)
        
        # Check overlap with already selected tiles
        ok_to_add = True
        for existing_box in selected_boxes:
            overlap_frac = compute_overlap(new_box, existing_box, tile_size)
            if overlap_frac > max_overlap:
                ok_to_add = False
                break
        
        if ok_to_add:
            selected.append(candidate)
            selected_boxes.append(new_box)
    
    print(f"Selected {len(selected)} tiles (requested: {n_tiles})")
    
    # If we didn't get enough tiles, relax overlap constraint
    if len(selected) < n_tiles:
        print(f"Relaxing overlap constraint to get {n_tiles} tiles...")
        selected = candidates[:n_tiles]
    
    # Step 3: Extract and prepare tiles
    tiles = []
    positions = []
    coverages = []
    scores = []
    
    for tile_info in selected:
        y, x = tile_info['position']
        
        # Extract regions
        m0_tile = moment0[y:y+tile_size, x:x+tile_size]
        m1_tile = moment1[y:y+tile_size, x:x+tile_size]
        m2_tile = moment2[y:y+tile_size, x:x+tile_size]
        
        # Prepare channels (same as before)
        moment0_clean = np.nan_to_num(m0_tile, nan=0.0)
        moment1_clean = np.nan_to_num(m1_tile, nan=0.0)
        moment2_clean = np.nan_to_num(m2_tile, nan=0.0)
        
        channel0 = moment0_clean
        channel1 = moment1_clean * moment0_clean
        channel2 = moment2_clean**2
        
        model_input = np.stack([channel0, channel1, channel2], axis=0)
        
        # Normalize
        for i in range(3):
            channel = model_input[i]
            if np.std(channel) > 0:
                model_input[i] = (channel - np.mean(channel)) / np.std(channel)
        
        tiles.append(model_input)
        positions.append(tile_info['position'])
        coverages.append(tile_info['coverage'])
        scores.append(tile_info['score'])
    
    # Convert to arrays
    tiles = np.array(tiles)
    positions = np.array(positions)
    coverages = np.array(coverages)
    scores = np.array(scores)
    
    print(f"\nTiles prepared:")
    print(f"  Shape: {tiles.shape}")
    print(f"  Coverage range: [{coverages.min():.1%}, {coverages.max():.1%}]")
    print(f"  Mean coverage: {coverages.mean():.1%}")
    
    # Print tile positions for verification
    print(f"\nTile positions (y, x):")
    for i, (pos, cov) in enumerate(zip(positions, coverages)):
        print(f"  Tile {i+1}: {pos} (coverage: {cov:.1%})")
    
    return {
        'tiles': tiles,
        'positions': positions,
        'coverages': coverages,
        'scores': scores,
        'original_shape': (ny, nx),
        'tile_size': tile_size,
        'n_tiles': len(tiles),
        'res_factors': np.array([tile_res_factor])
    }


def compute_overlap(box1, box2, tile_size):
    """
    Compute overlap fraction between two bounding boxes.
    
    Args:
        box1, box2: (y1, x1, y2, x2) tuples
        tile_size: Size of tiles for normalization
    
    Returns:
        overlap_fraction: Fraction of box area overlapping (0-1)
    """
    y1_1, x1_1, y2_1, x2_1 = box1
    y1_2, x1_2, y2_2, x2_2 = box2
    
    # Find intersection
    y1_int = max(y1_1, y1_2)
    x1_int = max(x1_1, x1_2)
    y2_int = min(y2_1, y2_2)
    x2_int = min(x2_1, x2_2)
    
    # Check if there's any intersection
    if y2_int <= y1_int or x2_int <= x1_int:
        return 0.0
    
    # Compute overlap area
    overlap_area = (y2_int - y1_int) * (x2_int - x1_int)
    tile_area = tile_size * tile_size
    
    return overlap_area / tile_area

def visualize_processing_pipeline(moment0, moment1, moment2, model_input, 
                                   emission_mask, output_path='perseus_processing.png'):
    """
    Create comprehensive visualization of data processing pipeline.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Raw moment maps
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(moment0, origin='lower', cmap='viridis')
    ax1.set_title('Moment 0: Integrated Intensity\n[K km/s]', fontsize=11)
    ax1.set_xlabel('X [pixels]')
    ax1.set_ylabel('Y [pixels]')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    # Mask moment1 for better visualization
    moment1_masked = np.where(emission_mask, moment1, np.nan)
    im2 = ax2.imshow(moment1_masked, origin='lower', cmap='RdBu_r')
    ax2.set_title('Moment 1: Velocity Centroid\n[km/s]', fontsize=11)
    ax2.set_xlabel('X [pixels]')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    moment2_masked = np.where(emission_mask, moment2, np.nan)
    im3 = ax3.imshow(moment2_masked, origin='lower', cmap='plasma')
    ax3.set_title('Moment 2: Velocity Dispersion\n[km/s]', fontsize=11)
    ax3.set_xlabel('X [pixels]')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(emission_mask, origin='lower', cmap='binary')
    ax4.set_title('Emission Mask\n(>3σ threshold)', fontsize=11)
    ax4.set_xlabel('X [pixels]')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Model input channels (128x128)
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(model_input[0], origin='lower', cmap='viridis')
    ax5.set_title('Channel 0: Column Density\n(normalized)', fontsize=11)
    ax5.set_xlabel('X [pixels]')
    ax5.set_ylabel('Y [pixels]')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(model_input[1], origin='lower', cmap='RdBu_r')
    ax6.set_title('Channel 1: Velocity-weighted\n(normalized)', fontsize=11)
    ax6.set_xlabel('X [pixels]')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(model_input[2], origin='lower', cmap='plasma')
    ax7.set_title('Channel 2: Dispersion\n(normalized)', fontsize=11)
    ax7.set_xlabel('X [pixels]')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(np.sqrt(model_input[0]**2 + model_input[1]**2 + model_input[2]**2),
                     origin='lower', cmap='magma')
    ax8.set_title('Combined Signal\n(L2 norm)', fontsize=11)
    ax8.set_xlabel('X [pixels]')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Row 3: Histograms
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.hist(model_input[0].flatten(), bins=50, alpha=0.7, color='green')
    ax9.set_xlabel('Normalized Value')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Channel 0 Distribution', fontsize=11)
    ax9.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(model_input[1].flatten(), bins=50, alpha=0.7, color='blue')
    ax10.set_xlabel('Normalized Value')
    ax10.set_ylabel('Frequency')
    ax10.set_title('Channel 1 Distribution', fontsize=11)
    ax10.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.hist(model_input[2].flatten(), bins=50, alpha=0.7, color='red')
    ax11.set_xlabel('Normalized Value')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Channel 2 Distribution', fontsize=11)
    ax11.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    # Statistics text
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    stats_text = f"""
    Processing Statistics:
    
    Input Shape: {moment0.shape}
    Output Shape: {model_input.shape}
    
    Emission Coverage:
    {100*emission_mask.sum()/emission_mask.size:.1f}% of pixels
    
    Channel 0 (Density):
    Mean: {model_input[0].mean():.3f}
    Std: {model_input[0].std():.3f}
    
    Channel 1 (Velocity):
    Mean: {model_input[1].mean():.3f}
    Std: {model_input[1].std():.3f}
    
    Channel 2 (Dispersion):
    Mean: {model_input[2].mean():.3f}
    Std: {model_input[2].std():.3f}
    """
    ax12.text(0.1, 0.5, stats_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Molecular Cloud: PPV Cube to Model Input Pipeline',
                fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    plt.close()

def process_ppvfits_to_model_input(fits_path, data_output_dir, mc_name='perseus', output_dir='./perseus_output', 
                                   data_format='auto', use_tiling=False, n_tiles=5, min_coverage=0.1, max_overlap=0.1):
    """
    Complete pipeline with optional tiling support.
    
    Args:
        fits_path: Path to FITS file
        mc_name: Name for output files
        output_dir: Directory for outputs
        data_format: 'auto', 'perseus', or 'taurus'
        use_tiling: If True, create tiles instead of single image
        n_tiles: Number of tiles to create (if use_tiling=True)
    
    Returns:
        model_input or tile_data depending on use_tiling
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    
    print("Load PPV Cube")
    data, header, wcs, velocity_axis = load_ppv_cube(fits_path, data_format=data_format)
    
    print("Compute Moment Maps")
    moment0, moment1, moment2, emission_mask = compute_moment_maps(data, velocity_axis)
    
    ny, nx = moment0.shape
    was_upsampled = min(ny, nx) < 128
    original_min_size = min(ny, nx)
    if min(ny, nx) < 128:
        print("Small cloud detected - upsampling moment maps")
        moment0, moment1, moment2, emission_mask = upsample_moment_maps(
            moment0, moment1, moment2, emission_mask, target_size=128
        )

    print("Physical Unit Conversion")
    column_density = convert_to_column_density(moment0, transition='13CO')
    
    if use_tiling:
        print("Prepare Tiled Model Input")
        tile_data = prepare_tiled_model_input(
            moment0, moment1, moment2, emission_mask, 
            n_tiles=n_tiles,  
            min_coverage=min_coverage,   
            max_overlap=max_overlap, was_upsampled = was_upsampled, original_min_size = original_min_size,
        )
        
        # Save tiles
        output_file = f'{data_output_dir}/{mc_name}_tiles.npz'
        np.savez(output_file, **tile_data)
        print(f"\nTiles saved: {output_file}")
        
        # Still save moment maps for reference
        np.savez(f'{data_output_dir}/{mc_name}_moments.npz',
                moment0=moment0, moment1=moment1, moment2=moment2,
                emission_mask=emission_mask, column_density=column_density,
                velocity_axis=velocity_axis)
        print(f"Moment maps saved: {data_output_dir}/{mc_name}_moments.npz")
        
        print("PROCESSING COMPLETE")
        return tile_data
    else:
        print("Prepare Model Input")
        model_input, res_factor = prepare_model_input(moment0, moment1, moment2, emission_mask, target_size=128)
        
        print("Visualization")
        visualize_processing_pipeline(moment0, moment1, moment2, model_input, 
                                    emission_mask, 
                                    output_path=f'{output_dir}/{mc_name}_processing.png')
        
        #Save outputs
        output_file = f'{data_output_dir}/{mc_name}_model_input.npy'
        np.save(output_file, model_input)
        np.save(output_file.replace('.npy', '_resfactor.npy'), np.array([res_factor]))
        print(f"\nModel input saved: {output_file}")
        
        # Save moment maps for reference
        np.savez(f'{data_output_dir}/{mc_name}_moments.npz',
                moment0=moment0, moment1=moment1, moment2=moment2,
                emission_mask=emission_mask, column_density=column_density,
                velocity_axis=velocity_axis)
        print(f"Moment maps saved: {data_output_dir}/{mc_name}_moments.npz")
        
        print("PROCESSING COMPLETE")
        print(f"\nReady for model inference:")
        print(f"  model_input.shape = {model_input.shape}")
        print(f"  Load with: data = np.load('{output_file}')")
        print(f"  Convert to tensor: torch.from_numpy(data).unsqueeze(0).float()")
    
        return model_input


if __name__ == "__main__":
    #Configuration
    PERSEUS_13CO_PATH = '/home/x-nbisht1/projects/p79d_dataset/observations/perseus/'
    PERSEUS_13CO_MOMENT_PATH = PERSEUS_13CO_PATH + 'PerA_13coFCRAO_F_map.fits.gz'
    PERSEUS_13CO_PPV_PATH = PERSEUS_13CO_PATH + 'PerA_13coFCRAO_F_xyv.fits.gz'
    PERSEUS_OUTPUT = '/home/x-nbisht1/plots/perseus/'

    PERSEUS_CENTER = SkyCoord('03h37m00s', '+31d49m00s', frame='icrs')
    PERSEUS_SIZE = 6.0 * u.degree
    OUTPUT_SIZE = 128

    TAURUS_13CO_PATH = '/home/x-nbisht1/projects/p79d_dataset/observations/taurus/'
    TAURUS_13CO_MOMENT_PATH = TAURUS_13CO_PATH + 'DHT21_Taurus_mom.fits'
    TAURUS_13CO_PPV_PATH = TAURUS_13CO_PATH + 'DHT21_Taurus_interp.fits'
    TAURUS_OUTPUT = '/home/x-nbisht1/plots/taurus/'

    SERPENS_13CO_PATH = '/home/x-nbisht1/projects/p79d_dataset/observations/serpens/'
    SERPENS_13CO_PPV_PATH = SERPENS_13CO_PATH + 'serpens_13co_xyv.fit.gz'#'13co10.fits'
    SERPENS_OUTPUT = '/home/x-nbisht1/plots/serpens/'

    ORION_13CO_PATH = '/home/x-nbisht1/projects/p79d_dataset/observations/orion/'
    ORION_13CO_PPV_PATH = ORION_13CO_PATH + 'Fig1.fits.gz' #Fig3a
    ORION_OUTPUT = '/home/x-nbisht1/plots/orion/'

    OPH_13CO_PATH = '/home/x-nbisht1/projects/p79d_dataset/observations/ophiucus/'
    OPH_13CO_PPV_PATH = OPH_13CO_PATH + 'OphA_13coFCRAO_F_xyv.fits.gz'
    OPH_OUTPUT = '/home/x-nbisht1/plots/ophiucus/'
    # Physical constants
    M_H2 = 2.8 * u.Da  # Mean molecular weight including He
    K_B = 1.380649e-23 * u.J / u.K
    # Process with tiling
    tile_data = process_ppvfits_to_model_input(fits_path=PERSEUS_13CO_PPV_PATH, data_output_dir = PERSEUS_13CO_PATH, mc_name='perseus',output_dir=PERSEUS_OUTPUT,data_format='auto',use_tiling=True,n_tiles=6,min_coverage=0.1, max_overlap=0.1)
    tile_data = process_ppvfits_to_model_input(fits_path=TAURUS_13CO_PPV_PATH, data_output_dir = TAURUS_13CO_PATH, mc_name='taurus',output_dir=TAURUS_OUTPUT,data_format='auto',use_tiling=True,n_tiles=2,min_coverage=0, max_overlap=0)
    tile_data = process_ppvfits_to_model_input(fits_path=SERPENS_13CO_PPV_PATH, data_output_dir = SERPENS_13CO_PATH, mc_name='serpens',output_dir=SERPENS_OUTPUT,data_format='auto',use_tiling=False,n_tiles=4,min_coverage=0, max_overlap=0)
    tile_data = process_ppvfits_to_model_input(fits_path=ORION_13CO_PPV_PATH, data_output_dir = ORION_13CO_PATH, mc_name='orion',output_dir=ORION_OUTPUT,data_format='auto',use_tiling=True,n_tiles=4,min_coverage=0, max_overlap=0)
    tile_data = process_ppvfits_to_model_input(fits_path=OPH_13CO_PPV_PATH, data_output_dir = OPH_13CO_PATH, mc_name='ophiucus',output_dir=OPH_OUTPUT,data_format='auto',use_tiling=True,n_tiles=4,min_coverage=0, max_overlap=0)
    