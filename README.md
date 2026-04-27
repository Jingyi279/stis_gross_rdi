G.R.O.S.S.: Great Resources for Occulted STIS Spectroscopy
=====================================
or, Get Rid Of STIS Spectroscopic Starlight

Tools for preparing STIS Coronagraphic Spectroscopy

The README lags behind the example notebook in the `example_notebooks` folder.

Observing mode overview
-----------------------

GROSS - a tool for preparing and analyzing data
-----------------------------------------------

The code is organized into three broad categories:

- Data preparation
- PSF modeling and subtraction
- Signal injection and retrieval analysis

### Data preparation

Data preparation involves:

- Applying calstis pipeline processing, if necessary.
- Cleaning cosmics missed by the cosmic-ray rejection algorithm.
- Cleaning hot and cold pixels in the data.
- Defringing 2-D spectral images
- Organizing the files in a logical way. Typically this means grouping the
  following files for a single target:
  - 1-D extracted spectrum
  - 2-D unocculted spectral image
  - 2-D occulted spectral image



- `ObservingSequence` class
  - three arguments (same as above)
    - extracted 1-D spectrum
    - 2-D unocculted spectral image
    - 2-D occulted spectral image
  
  `G.R.O.S.S.` uses `ObservingSequence` objects to keep related files together.
  Specifically, some observing sequence includes a TA image, an unocculted
  exposure, and an occulted exposure. The hstcal pipeline also generates a 1-D
  spectrum from the unocculted trace that is useful in processing the exposures.
  ObservingSequence objects use these files to derive the row position of the
  star in the occulted exposures and prepare the PSF subtraction. 
  
  - Attributes:
    - `_files` : a dict that keeps a record of the initializing files
    - `wlsol`: the wavelength solution, in meters
    - `primary_spectrum_counts`: the point source spectrum, in units of counts/sec
    - `primary_spectrum_counts_unc`: the associated uncertainty from the `ERR` column
    - `unocc_wcs`: WCS object for the unocculted observation
    - `unocc_img`: 2-D spectral image of the unocculted
    - `offset`: distance in degrees of the unocculted point source from the nominal position
    - `unocc_primary_row`: 0-indexed row coordinate corresponding to the offset in the unocculted exposure
    - `unocc_trace`: astropy.nddata.Cutout2D crop of the spectral trace; used for injection and recovery tests
    - `occ_wcs`: WCS object for the occulted observation
    - `occ_img`: 2-D spectral image of the occulted observation
    - `occ_row`: 0-indexed row coordinate corresponding to the offset in the occulted exposure, aka where the star is

- `coronspec_tools::find_star::find_star_from_wcs()`
  - three arguments:
    - extracted 1-D spectrum
    - 2-D unocculted spectral image
    - 2-D occulted spectral image

    The observing sequence for this mode is to take an unocculted exposure at the F1
    position and then an occulted exposure at the E1 position, along the slit. This
    function uses the WCS information to find the position of the star in the
    occulted image. It assumes that the HST slews are very accurate. It measures the
    difference between the actual location of the trace in the unocculted 2-D image,
    and the nominal position of the trace in the WCS at coordinates (wl_min, 0 deg).
    Assuming the offset is preserved in the slew between fiducial positions, it
    applies the measured offset to the nominal position in the occulted exposure.

### PSF modeling and subtraction

- `sdi_tools.py` 
  This module contains most of the tools needed for spectral differential
  imaging. The current strategy is to rescale the image by wavelength so that
  the speckles go straight down the rows and an off-axis PSF follows a 1/lambda
  curve. For each row in the scaled image, the columns at which the PSF (real or
  hypothetical) crosses the row are masked out, and the value of the speckle is
  replaced by interpolation to construct a model of the stellar PSF. This model
  is then subtracted from the scaled image to produce a residual image. The
  signal along the off-axis PSF path can be projected back onto the original,
  descaled image to construct a residual map for every row.


### Signal injection, detection, and retrieval
