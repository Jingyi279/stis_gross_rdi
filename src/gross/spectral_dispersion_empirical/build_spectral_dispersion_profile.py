import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, ImageNormalize, SqrtStretch)
from scipy.optimize import curve_fit, differential_evolution
from scipy.special import gamma

import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from astropy import wcs
import warnings


# load the reference datasets 

# occulted star behind the bar
with fits.open('of0i04010_sx2.fits') as hdul:
    hdul.info()
    unocc_sci = hdul[1].data
    unocc_sci_hdr = hdul[1].header

# same star, but unocculted shorter exposure
with fits.open('of0i04020_sx2.fits') as hdul:
    hdul.info()
    occ_sci = hdul[1].data
    occ_sci_hdr = hdul[1].header    


if display_plots == True:
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(unocc_sci)
    plt.imshow(unocc_sci, vmin=vmin, vmax=vmax, origin='lower')
    plt.title('Unocculted Reference Star Spectra')
    plt.show()

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(occ_sci)
    plt.imshow(occ_sci, vmin=vmin, vmax=vmax, origin='lower')
    plt.title('Occulted Reference Star Spectra')
    plt.show()

    # try to measure the PSF wings at a specific wavelength in both spectra (let's do above bar)
    plt.plot(occ_sci[:,400])
    plt.xlim(500,650)
    plt.show()

    # plot the unocculted trace for reference
    plt.plot(unocc_sci[:,400], '.-', ms=5)
    plt.xlim(950,1050)


# ------ fitting tools ----- #
# make a modified curve fitting function as an experiment
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)

def linear(x, slope, intercept):
    return slope * x + intercept



def moffat(x, amplitude, center, alpha, beta):
    return amplitude * (1 + ((x - center) / alpha)**2) ** (-beta)


def fit_line(wavelength, flux, line_center, window=0.1, continuum_window=0.05):
    """
    Fit a Moffat + linear continuum to a spatial trace profile.

    Parameters
    ----------
    wavelength       : pixel array along the spatial (cross-dispersion) axis
    flux             : flux array corresponding to wavelength
    line_center      : approximate trace center (pixels)
    window           : half-width of region to fit (px)
    continuum_window : width of continuum regions on each side (px)

    Returns
    -------
    dict with center, alpha, beta, fwhm, integrated flux, and continuum-subtracted spectrum
    """
    # Clip to fitting window
    mask = np.abs(wavelength - line_center) <= window
    wv   = wavelength[mask]
    fl   = flux[mask]

    # Fit continuum from flanking regions on either side of the line
    cont_mask = (np.abs(wavelength - line_center) > window - continuum_window) & mask
    cont_wv   = wavelength[cont_mask]
    cont_fl   = flux[cont_mask]
    cont_coeffs = np.polyfit(cont_wv, cont_fl, 1)
    continuum   = np.polyval(cont_coeffs, wv)
    flux_sub    = fl - continuum

    # Initial guesses
    amp0    = np.max(flux_sub)
    center0 = np.sum(wv * flux_sub) / np.sum(flux_sub)   # flux-weighted centroid

    # --- Stage 1: differential evolution for robust global solution ---
    def chi2(params):
        amp, cen, alp, bet = params
        if amp <= 0 or alp <= 0 or bet <= 0.5:
            return 1e30
        return np.sum((flux_sub - moffat(wv, amp, cen, alp, bet))**2)

    bounds_de = [(0,              amp0 * 3      ),
                 (center0 - 1.0, center0 + 1.0  ),
                 (0.1,           window          ),
                 (0.5,           15.0            )]

    de_result = differential_evolution(chi2, bounds_de, seed=42,
                                       maxiter=10000, tol=1e-14)

    # --- Stage 2: refine with curve_fit/trf starting from DE solution ---
    popt, pcov = curve_fit(moffat, wv, flux_sub, p0=de_result.x,
                           method='trf', maxfev=100000)
    perr = np.sqrt(np.diag(pcov))

    amplitude, center, alpha, beta = popt
    amp_err, center_err, alpha_err, beta_err = perr

    # Integrated flux = amplitude * alpha * sqrt(π) * Γ(β - 0.5) / Γ(β)
    # (valid for β > 0.5, which our DE lower bound ensures)
    gamma_ratio         = gamma(beta - 0.5) / gamma(beta)
    integrated_flux     = amplitude * alpha * np.sqrt(np.pi) * gamma_ratio

    # Partial derivative of gamma ratio w.r.t. beta via finite difference
    db = 1e-5
    dgamma_ratio_dbeta  = (gamma(beta - 0.5 + db) / gamma(beta + db) -
                           gamma(beta - 0.5 - db) / gamma(beta - db)) / (2 * db)
    integrated_flux_err = integrated_flux * np.sqrt(
        (amp_err   / amplitude)                              ** 2 +
        (alpha_err / alpha)                                  ** 2 +
        (dgamma_ratio_dbeta * beta_err / gamma_ratio)        ** 2
    )

    # FWHM of a Moffat: 2 * alpha * sqrt(2^(1/beta) - 1)
    fwhm = 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)

    # Upsampled model grid
    wv_model = np.linspace(np.min(wv), np.max(wv), 10000)

    return {
        'center':              center,
        'center_err':          center_err,
        'alpha':               alpha,
        'alpha_err':           alpha_err,
        'beta':                beta,
        'beta_err':            beta_err,
        'fwhm':                fwhm,
        'integrated_flux':     integrated_flux,
        'integrated_flux_err': integrated_flux_err,
        'wv':                  wv,
        'wv_model':            wv_model,
        'flux_sub':            flux_sub,
        'continuum':           continuum,
        'fit':                 moffat(wv_model, *popt),
    }




def moffat_fixed_shape(x, amplitude, center, alpha, beta):
    return amplitude * (1 + ((x - center) / alpha)**2) ** (-beta)

def fit_amplitude_only(yrange, flux_col, center_guess, alpha_fixed, beta_fixed,
                       window=15, continuum_window=2):
    """
    Fit only amplitude and center, with alpha and beta held fixed.
    """
    mask = np.abs(yrange - center_guess) <= window
    wv   = yrange[mask]
    fl   = flux_col[mask]

    cont_mask   = (np.abs(yrange - center_guess) > window - continuum_window) & mask
    cont_coeffs = np.polyfit(yrange[cont_mask], flux_col[cont_mask], 1)
    continuum   = np.polyval(cont_coeffs, wv)
    flux_sub    = fl - continuum

    # wrap moffat with alpha and beta frozen
    def moffat_2param(x, amplitude, center):
        return moffat_fixed_shape(x, amplitude, center, alpha_fixed, beta_fixed)

    amp0    = np.max(flux_sub)
    center0 = np.sum(wv * flux_sub) / np.sum(flux_sub)

    popt, pcov = curve_fit(moffat_2param, wv, flux_sub,
                           p0=[amp0, center0],
                           bounds=([0,          center0 - 1.0],
                                   [np.inf,     center0 + 1.0]),
                           method='trf', maxfev=100000)
    perr = np.sqrt(np.diag(pcov))

    amplitude, center = popt
    alpha, beta = alpha_fixed, beta_fixed

    return {
        'amplitude':   amplitude,
        'amplitude_err': perr[0],
        'center':      center,
        'center_err':  perr[1],
        'alpha':       alpha,
        'beta':        beta,
        'fwhm':        2 * alpha * np.sqrt(2**(1/beta) - 1),
        'flux_sub':    flux_sub,
        'wv':          wv,
        'continuum':   continuum,
    }

# old approach -- Gaussian
# def fit_line(wavelength, flux, line_center, window=0.1, continuum_window=0.05):
#     """
#     Fit a Gaussian + linear continuum to a trace.
    
#     Parameters
#     ----------
#     wavelength    : pixel array around line of interest (not wavelength, since we're along the spatial y-axis)
#     flux          : full flux array
#     line_center   : approximate trace center 
#     window        : half-width of region to fit (px)
#     continuum_window : width of continuum regions on each side (px)
    
#     Returns
#     -------
#     dict with center, width (sigma), integrated flux, and continuum-subtracted spectrum
#     """
#     # Clip to fitting window
#     mask = np.abs(wavelength - line_center) <= window
#     wv   = wavelength[mask]
#     fl   = flux[mask]

#     # Fit continuum from flanking regions on either side of the line
#     cont_mask = (np.abs(wavelength - line_center) > window - continuum_window) & mask
#     cont_wv   = wavelength[cont_mask]
#     cont_fl   = flux[cont_mask]

#     cont_coeffs      = np.polyfit(cont_wv, cont_fl, 1)
#     continuum        = np.polyval(cont_coeffs, wv)
#     flux_sub         = fl - continuum

#     # Initial guesses
#     # More targeted initial guesses
#     amp0    = np.max(flux_sub)
#     print(f'amp0 is {amp0}')
#     center0 = wv[np.argmax(flux_sub)] 
#     sigma0  = 2


#     p0     = [amp0, center0, sigma0]
#     bounds = ([0,    line_center - window, 0      ],
#               [amp0*1.50, line_center + window, window-7]) # allow amplitude to be up to 50% larger, but that's it 

#     popt, pcov = curve_fit(gaussian, wv, flux_sub, p0=p0, bounds=bounds, max_nfev=100000)
#     perr       = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

#     amplitude, center, sigma = popt
#     amp_err, center_err, sigma_err = perr

#     # Integrated flux = amplitude * sigma * sqrt(2π)
#     integrated_flux     = amplitude * sigma * np.sqrt(2 * np.pi)
#     integrated_flux_err = integrated_flux * np.sqrt((amp_err/amplitude)**2 + (sigma_err/sigma)**2)

#     # upsampled wavelength grid
#     wv_model = np.linspace(np.min(wv), np.max(wv), 10000)

#     return {
#         'center':              center,
#         'center_err':          center_err,
#         'sigma':               sigma,
#         'sigma_err':           sigma_err,
#         'fwhm':                2.355 * sigma,
#         'integrated_flux':     integrated_flux,
#         'integrated_flux_err': integrated_flux_err,
#         'wv':                  wv,
#         'wv_model':            wv_model,
#         'flux_sub':            flux_sub,
#         'continuum':           continuum,
#         'fit':                 gaussian(wv_model, *popt),
#     }


# ---- try example ---- #

window = 15
trace_center = 985
xcol_value = 400

# define yrange
yrange = np.arange(len(unocc_sci[:,xcol_value]))
if do_example == True:
# fit_line(wavelength, flux, line_center, window=0.1, continuum_window=0.05)
    fit_example = fit_line(yrange, unocc_sci[:,xcol_value], line_center = trace_center, window = window, continuum_window = 2)
    yrange = np.arange(len(unocc_sci[:,xcol_value]))
    plt.plot(yrange, unocc_sci[:,xcol_value])
    plt.xlim(977,992)
    plt.vlines(fit_example['center'], 0, np.max(unocc_sci[:,xcol_value]))
    plt.plot(fit_example['wv_model'], fit_example['fit'])



# ---- load object spectrum to scale trace by ---- #
target_spectrum = np.genfromtxt('./accreting_obj_G750L-resolution.txt').T
target_spectrum_flux = target_spectrum[1]

def build_spectral_profile(data, yrange, trace_center, target_spectrum,
                           half_window=10, bin_size=100, plot=False, plot_every=50):
    """
    Build a rescalable 2D spectral profile using a median master profile
    rather than a parametric fit. Robust for undersampled or asymmetric traces.

    Parameters
    ----------
    data            : 2D array (nrows, ncols)
    yrange          : 1D array of pixel positions along spatial axis
    trace_center    : approximate center row of the trace
    target_spectrum : 1D array (length ncols) — flux per column
    half_window     : rows on each side of trace_center to include
    bin_size        : columns to median-combine for master profile shape
    plot            : if True, plot data vs fit for selected columns
    plot_every      : plot every Nth column (default 50)

    Returns
    -------
    normalized_profile_2d : 2D array — profile normalized to sum=1 per column
    profile_2d            : 2D array — profile rescaled by target_spectrum
    fit_params            : dict of 1D arrays with 'amplitude', 'amplitude_err'
    """
    nrows, ncols = data.shape
    pmin = trace_center - half_window
    pmax = trace_center + half_window
    rows_in_window = yrange[pmin:pmax]

    profile_2d            = np.zeros((nrows, ncols))
    normalized_profile_2d = np.zeros((nrows, ncols))
    fit_params            = {'amplitude':     np.full(ncols, np.nan),
                             'amplitude_err': np.full(ncols, np.nan)}

    # --- stage 1: build master profile by median-combining binned columns ---
    nbins = ncols // bin_size
    bin_profiles = np.zeros((pmax - pmin, nbins))

    for j in range(nbins):
        bin_col = np.nanmedian(data[pmin:pmax, bin_size*j : bin_size*j + bin_size], axis=1)
        bg = np.nanmedian(np.concatenate([bin_col[:2], bin_col[-2:]]))
        bin_col_sub = np.maximum(bin_col - bg, 0)
        bin_sum = np.sum(bin_col_sub)
        if bin_sum > 0:
            bin_profiles[:, j] = bin_col_sub / bin_sum

    master_profile_window = np.nanmedian(bin_profiles, axis=1)
    master_sum = np.sum(master_profile_window)
    if master_sum <= 0:
        raise ValueError("Master profile sums to zero — check trace_center and half_window")
    master_profile_norm = master_profile_window / master_sum

    print(f"Master profile built from rows {pmin}:{pmax}")
    print(f"Peak row in window: {pmin + np.argmax(master_profile_norm)}")
    print(f"Normalized peak value: {np.max(master_profile_norm):.4f}")

    # --- stage 2: per-column least-squares amplitude ---
    for xcol in range(ncols):
        try:
            col_data = data[pmin:pmax, xcol].astype(float)

            bg = np.nanmedian(np.concatenate([col_data[:2], col_data[-2:]]))
            col_sub = col_data - bg

            denom = np.sum(master_profile_norm**2)
            A     = np.sum(col_sub * master_profile_norm) / denom

            residuals  = col_sub - A * master_profile_norm
            rms        = np.sqrt(np.mean(residuals**2))
            A_err      = rms / np.sqrt(denom)

            fit_params['amplitude'][xcol]     = A
            fit_params['amplitude_err'][xcol] = A_err

            full_profile = np.zeros(nrows)
            full_profile[pmin:pmax] = master_profile_norm

            normalized_profile_2d[:, xcol] = full_profile
            profile_2d[:, xcol]            = full_profile * target_spectrum[xcol]

            # --- diagnostic plot ---
            if plot and (xcol % plot_every == 0):
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f"Column {xcol}", fontsize=12)

                # left panel: data vs fit in the window
                fit_in_window = A * master_profile_norm + bg
                axes[0].plot(rows_in_window, col_data,
                             'k.', ms=6, label='data')
                axes[0].plot(rows_in_window, fit_in_window,
                             'r-', lw=1.5, label=f'fit  (A={A:.3e})')
                axes[0].axhline(bg, color='gray', lw=1, ls='--', label='background')
                axes[0].set_xlabel('row (px)')
                axes[0].set_ylabel('flux')
                axes[0].set_title('trace window')
                axes[0].legend(fontsize=8)

                # right panel: residuals
                axes[1].plot(rows_in_window, residuals,
                             'k.', ms=6, label='residuals')
                axes[1].axhline(0, color='r', lw=1, ls='--')
                axes[1].axhline( rms, color='gray', lw=1, ls=':', label=f'±rms={rms:.2e}')
                axes[1].axhline(-rms, color='gray', lw=1, ls=':')
                axes[1].set_xlabel('row (px)')
                axes[1].set_ylabel('data - fit')
                axes[1].set_title('residuals')
                axes[1].legend(fontsize=8)

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"  column {xcol} failed: {e}")

    n_good = np.sum(np.isfinite(fit_params['amplitude']))
    print(f"Done. {n_good}/{ncols} columns completed.")
    return normalized_profile_2d, profile_2d, fit_params


# generate a fit on existing data
norm_2d, target_trace_2d, fit_params = build_spectral_profile(
    unocc_sci, yrange, trace_center=984,
    target_spectrum=target_spectrum_flux,
    half_window=10, bin_size=100,
    plot=True, plot_every=50
)

# display trace?
if display_trace == True:
    # scaled trace
    z = ZScaleInterval()
    z1,z2 = z.get_limits(target_trace_2d)
    plt.imshow(target_trace_2d, origin='lower')#, vmin = z1, vmax = z2)
    fits.writeto('test_trace.fits', target_trace_2d, overwrite=True)

    # normalized template trace
    z = ZScaleInterval()
    z1,z2 = z.get_limits(norm_2d)
    plt.imshow(norm_2d, origin='lower')#, vmin = z1, vmax = z2)
    fits.writeto('test_norm.fits', norm_2d, overwrite=True)    

