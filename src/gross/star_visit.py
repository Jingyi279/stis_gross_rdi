"""
Tools for managing an observing sequence
"""
from pathlib import Path
import warnings

import numpy as np


from astropy import units
from astropy import stats
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS, FITSFixedWarning
from astropy.utils.exceptions import AstropyWarning

from coronspec_tools import utils as ctutils
from coronspec_tools import find_star as ctfs

# the WCS throws an warning when loading this data that we can ignore
warnings.filterwarnings("ignore", category=FITSFixedWarning)

class StarVisit:

    def __init__(
            self,
            sx1_file : str | Path,
            unocc_file : str | Path,
            occ_file : str | Path,
            trace_width : int = 11,
            occ_stamp_width : int = 61,
            median_clean : int = 10,
            contrast : bool = False,
    ) -> None :
        """
        Instantiate a class to manage injection of spectral traces into 2-D
        spectral images. This includes rescaling and reshaping the spectra.

        Parameters
        ----------
        sx1_file : str | Path
          file containing the wavelength solution and extracted spectrum
        unocc_file : str | Path
          file containing the 2-D unocculted spectral image
        occ_file : str | Path
          file containing the 2-D occulted spectral image
        trace_width : int = 11
          the width of the trace image to cut out, for PSF injection
        occ_stamp_width : int = 61
          the width of the occulted image to cut out for PSF subtraction
        median_clean : int = 10
          Apply a rolling median along the wavelength axis with this width to
          clean the data. If >= 0, not applied.
        contrast : bool = False
          if True, convert the occ and unocc images to counts/sec, and divide
          by the primary unocculted spectrum

        Output
        ------
        None

        """
        self._files = {'sx1': sx1_file, 'unocc': unocc_file, 'occ': occ_file}
        self.hdrs = {
            k: {h: fits.getheader(v, h) for h in [0,'sci']}
            for k, v in self._files.items()
        }
        # pull data out of the files
        # spectral information
        self.process_specfile(sx1_file)
        # unocculted
        self.process_unocculted(unocc_file, trace_width)
        # occulted
        self.process_occulted(occ_file)

        # process data
        self.occ_primary_row, self.occ_bar = ctfs.find_star_from_wcs(
            sx1_file, unocc_file, occ_file,
        )
        # if str(occ_file)[:-5].split("_")[-1] == 'sx2':
        #     self.occ_primary_row -= 88.5
        self.occ_sep = self.occ_wcs.pixel_to_world(
            0, self.occ_primary_row
        )[1]
        self.make_occ_stamp(occ_stamp_width)
        # data cleaning
        if median_clean > 0:
            specunit = self.primary_spectrum_counts.unit
            self.primary_spectrum_counts = ctutils.rolling_median(
                self.primary_spectrum_counts.value, median_clean
            ) * specunit
            self.primary_spectrum_counts_unc = ctutils.rolling_median(
                self.primary_spectrum__counts_unc.value, median_clean
            ) * specunit
            self.occ_stamp.data = self.clean_stamp(self.occ_stamp.data, median_clean)
            self.occ_stamp_unc.data = self.clean_stamp(self.occ_stamp_unc.data, median_clean)
            self.unocc_trace.data = self.clean_stamp(self.unocc_trace.data, median_clean)
        if contrast:
            # convert to units of contrast
            self.is_contrast = True
            self.convert_to_contrast()
        return None

    def make_occ_stamp(self, width):
        # make the occulted stamp and record the center row
        self.occ_stamp = Cutout2D(
            self.occ_img, 
            position=(self.occ_img.shape[1]/2, self.occ_primary_row),
            size=(width, self.occ_img.shape[1]),
            wcs=self.occ_wcs,
            copy=True,
        )
        self.occ_stamp_unc = Cutout2D(
                    self.occ_unc, 
                    position=(self.occ_unc.shape[1]/2, self.occ_primary_row),
                    size=(width, self.occ_unc.shape[1]),
                    wcs=self.occ_wcs,
                    copy=True,
                )
        self.occ_stamp_dq = Cutout2D(
                    self.occ_dq, 
                    position=(self.occ_dq.shape[1]/2, self.occ_primary_row),
                    size=(width, self.occ_dq.shape[1]),
                    wcs=self.occ_wcs,
                    copy=True,
                )
        self.occ_stamp_center = self.occ_primary_row - self.occ_stamp.origin_original[1]


    def process_specfile(self, specfile):
        with fits.open(specfile) as hdulist:
            table = hdulist[1].data
            colname = 'WAVELENGTH'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if colunit == 'Angstroms':
                colunit = 'Angstrom'
            self.wlsol = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            ).to(units.m)
            colname = 'NET'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_counts = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            colname = 'NET_ERROR'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_counts_unc = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            colname = 'FLUX'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_flux = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            colname = 'ERROR'
            colind = table.names.index(colname)+1
            colunit = hdulist[1].header[f'TUNIT{colind}']
            if 'Counts' in colunit:
                colunit = colunit.replace("Counts",'count')
            self.primary_spectrum_flux_unc = units.Quantity(
                np.squeeze(table[colname]),
                unit=colunit
            )
            self.spectral_response_function = self.primary_spectrum_flux / self.primary_spectrum_counts
            self.spectral_response_function_unc = self.spectral_response_function * np.sqrt(
                (self.primary_spectrum_flux_unc/self.primary_spectrum_flux)**2 +\
                (self.primary_spectrum_counts_unc/self.primary_spectrum_counts)**2
            )
        return

    def process_unocculted(self, unocc_file, trace_width):
        # measure the TA offset
        # get the wcs and unocculted image
        with fits.open(unocc_file) as hdulist:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                self.unocc_wcs = WCS(hdulist[1].header)
            self.unocc_img = hdulist['SCI'].data.copy()
            self.unocc_unc = hdulist['ERR'].data.copy()
            filetype = str(unocc_file)[:-5].split("_")[-1]
            self.offset, self.unocc_primary_row = ctfs.find_unocc_pos(hdulist, self.wlsol)
        self.unocc_trace = self.get_unocc_trace(trace_width)
        return

    def process_occulted(self, occ_file):
        # get the wcs and occulted image
        with fits.open(occ_file) as hdulist:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.occ_wcs = WCS(hdulist['SCI'].header)
            self.occ_img = hdulist['SCI'].data.copy()
            self.occ_unc = hdulist['ERR'].data.copy()
            self.occ_dq = hdulist['DQ'].data.copy()
            filetype = str(occ_file)[:-5].split("_")[-1]
            return

    def clean_stamp(self, img, width=10, std_thresh=100):
        """
        Median-filter the outlier pixels
        """
        # avg, std = stats.sigma_clipped_stats(self.occ_stamp.data)
        # thresh = avg + std_thresh*std
        """Apply filtering to a 2-D spectral image"""
        filtimg = ctutils.median_filter_image(img, width)
        # filtimg = np.array([
        #     ctutils.savgol_filter(row, width=width, order=3)
        # ])
        return filtimg
        


    def get_unocc_trace(self, trace_width):
        trace = Cutout2D(
            self.unocc_img, 
            position=(self.unocc_img.shape[1]/2, self.unocc_primary_row),
            size=(trace_width, self.unocc_img.shape[1]),
            wcs=self.unocc_wcs,
            copy=True,
        )
        return trace

    def convert_to_contrast(self) -> None:
        """Converts all relevant data to units of contrast"""
        spectrum = self.primary_spectrum_counts.value # counts/sec
        # first, convert to counts/sec
        exptime = self.hdrs['unocc']['sci']['exptime']
        # spectrum = self.unocc_trace.data.sum(axis=0) / exptime
        self.unocc_img = self.unocc_img / exptime / spectrum
        self.unocc_trace.data = self.unocc_trace.data / exptime / spectrum

        exptime = self.hdrs['occ']['sci']['exptime']
        self.occ_img = self.occ_img / exptime / spectrum
        self.occ_stamp.data = self.occ_stamp.data / exptime / spectrum

        return

   def contrast_counts2flux(self, signal):
        """Convert a row of signal from contrast units to flux"""
        return signal * self.spectral_response_function

    def get_bar_bounds(self) -> tuple[float, float]:
        halfwidth = (0.25*units.arcsec).to(units.deg)
        lb = self.occ_wcs.world_to_pixel(
            self.occ_wcs.pixel_to_world(0, self.occ_primary_row)[0],
            self.occ_wcs.pixel_to_world(0, self.occ_bar)[1] - halfwidth
        )[1]
        ub = self.occ_wcs.world_to_pixel(
            self.occ_wcs.pixel_to_world(0, self.occ_primary_row)[0],
            self.occ_wcs.pixel_to_world(0, self.occ_bar)[1] + halfwidth 
        )[1]
        return lb, ub

