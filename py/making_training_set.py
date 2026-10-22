import os

os.environ["STPSF_PATH"]="/blue/adamginsburg/t.yoo/from_red/stpsf-data"
os.environ["WEBBPSF_PATH"]="/blue/adamginsburg/t.yoo/from_red/stpsf-data"


import stpsf
import glob
from astropy.io import fits
from scipy.ndimage import label, find_objects, center_of_mass, sum_labels
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clipped_stats
from jwst.datamodels import dqflags
import matplotlib.pyplot as plt
try:
    # version >=1.7.0, doesn't work: the PSF is broken (https://github.com/astropy/photutils/issues/1580?)
    from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper
except:
    # version 1.6.0, which works
    from photutils.psf import BasicPSFPhotometry as PSFPhotometry, IterativelySubtractedPSFPhotometry as IterativePSFPhotometry, DAOGroup as SourceGrouper
try:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D, LocalBackground
except:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D
    from photutils.background import MMMBackground as LocalBackground
from jwst.datamodels import RampModel

from astroquery.svo_fps import SvoFps

from tqdm.notebook import tqdm
from tqdm import tqdm
from astropy import wcs
from astropy.wcs import WCS
import numpy as np
from scipy import ndimage
from astropy.table import Table, QTable
from astropy import table
from astropy import log
from filtering import get_filtername, get_fwhm
import functools
import sys
sys.path.insert(0, '/blue/adamginsburg/t.yoo/w51_miri/reduction')
import miri_4qpm  # MIRI coronagraph (4QPM) pane geometry
import requests
import urllib3
def get_psf(header, path_prefix='.'):
    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')
    instrument = header['INSTRUME']
    filtername = get_filtername(header)
    try:
        module = header['MODULE']
    except KeyError:
        module = header['DETECTOR']
    detector = header['DETECTOR']

    ww = wcs.WCS(header)
    try:
        assert ww.wcs.cdelt[1] != 1, "This is not a valid WCS!!! CDELT is wrong!! how did this HAPPEN!?!?"
    except AssertionError as ex:
        print(ex)
        print("ignoring WCS failure so check that stuff is right...")

    psfgen.filter = filtername
    obsdate = header['DATE-OBS']

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()

    npsf = 16
    oversample = 2
    fov_pixels = 512
    if detector == 'NRCALONG':
        detector = 'nrca5'
    elif detector == 'NRCBLONG':
        detector = 'nrcb5'
    if detector.lower() == 'mirimage':
        detector = 'mirim'
    psf_fn = f'{path_prefix}/{instrument.lower()}_{detector.lower()}_{filtername.lower()}_fovp{fov_pixels}_samp{oversample}_npsf{npsf}.fits'

    if module == 'merged':
        project_id = header['PROGRAM'][1:5]
        obs_id = header['OBSERVTN'].strip()
        merged_psf_fn = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid.fits'
        if os.path.exists(psf_fn):
            psf_fn = merged_psf_fn
        else:
            print("stpsf is being used for merged data because merged PSF does not exist", flush=True)

    if os.path.exists(str(psf_fn)):
        # As a file
        log.info(f"Loading grid from psf_fn={psf_fn}")
        big_grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
        if isinstance(big_grid, list):
            print(f"PSF IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
    else:
        log.info(f'PSF file {psf_fn} does not exist; downloading from MAST')
        from astroquery.mast import Mast

        print(f"Attempting to load PSF for {obsdate}")
        try:
            Mast.login(api_token.strip())
            os.environ['MAST_API_TOKEN'] = api_token.strip()

            psfgen.load_wss_opd_by_date(f'{obsdate}T00:00:00')
        except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
            print(f"Failed to build PSF: {ex}")
        except Exception as ex:
            print("psfgen load_wss_opd_by_date failed")
            print(ex)

        log.info(f"starfinding: Calculating grid for psf_fn={psf_fn}")
        # https://github.com/spacetelescope/stpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/stpsf/stpsf_core.py#L640
        # https://github.com/spacetelescope/stpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/stpsf/gridded_library.py#L424
        big_grid = psfgen.psf_grid(num_psfs=npsf, oversample=oversample,
                                   all_detectors=True, fov_pixels=fov_pixels,
                                   outdir=path_prefix,
                                   save=True, outfile=None, overwrite=True)
        # now the PSF should be written
        assert glob.glob(psf_fn.replace(".fits", "*"))
        if isinstance(big_grid, list):
            print(f"PSF FROM PSF_GEN IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
            # if we really want to get this right, we need to create a new grid of PSF models
            # that is some sort of average of the PSF model grid.
            # There's no way to do it _right_ right without going back to the original data,
            # which is untenable with this approach.  It's a huge project.

    return big_grid
def debug_wrap(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        print(function.__name__, flush=True)
        return function(*args, **kwargs)
    return wrapper

import os
#os.environ['stpsf_PATH'] = '/orange/adamginsburg/jwst/stpsf-data/'
import stpsf
from stpsf.utils import to_griddedpsfmodel



def collect_saturated_pixels(fitsdata, rampdata, path_prefix='/orange/adamginsburg/jwst/w51/psfs/', pad=30, size=None,
min_sep_from_edge=5, edge_npix=10000, mask_buffer=1, plot=True, rindsz=3, filtername=None, instrument=None, detector=None, fn=None, img_save_dir = '/orange/adamginsburg/w51/TaehwaYoo/w51_sat_recover/cutout_img',
area_ratio_threshold=8.0):
    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    jfilts = SvoFps.get_filter_list(facility="JWST", instrument=instrument)
    row = jfilts[jfilts["filterID"] == f"JWST/{instrument}.{filtername}"][0]
    filter_bandwidth = row["WidthEff"]
    header = fitsdata[0].header
    data = fitsdata['SCI'].data
    cal_value_map = data.copy()  # pipeline value per pixel, kept as an alternative label
    fqpm_quadrant_map = (miri_4qpm.pane_quadrant_image(line_halfwidth=0) if instrument == 'MIRI'
                         else np.zeros(data.shape, np.uint8))  # 0 = imager field, 1-12 = pane quadrant
    data[np.isnan(fitsdata['VAR_POISSON'].data)] = 0
    dq = fitsdata['DQ'].data
    groupdq = rampdata['GROUPDQ'].data

    
    

    results_sats = Table()

    #for isat in range(5):
     #   saturated = (groupdq[0,isat,:,:] & dqflags.group['SATURATED'])>0
    saturated = (dq & dqflags.pixel['SATURATED'])>0 
    issaturated_0 = (groupdq[0,0,:,:] & dqflags.group['SATURATED'])>0
    issaturated_1 = (groupdq[0,1,:,:] & dqflags.group['SATURATED'])>0
    issaturated_2 = (groupdq[0,2,:,:] & dqflags.group['SATURATED'])>0
    issaturated_3 = (groupdq[0,3,:,:] & dqflags.group['SATURATED'])>0
    issaturated_4 = (groupdq[0,4,:,:] & dqflags.group['SATURATED'])>0
    
    issaturated_4_only = issaturated_4 & ~issaturated_3 & ~issaturated_2 & ~issaturated_1 & ~issaturated_0
    issaturated_3_only = issaturated_3 & ~issaturated_2 & ~issaturated_1 & ~issaturated_0
    issaturated_2_only = issaturated_2 & ~issaturated_1 & ~issaturated_0
    issaturated_1_only = issaturated_1 & ~issaturated_0
    issaturated_0_only = issaturated_0

    sat_level_map = np.ones_like(saturated, dtype=int) * -1
    sat_level_map[issaturated_0_only] = 0
    sat_level_map[issaturated_1_only] = 1
    sat_level_map[issaturated_2_only] = 2
    sat_level_map[issaturated_3_only] = 3
    sat_level_map[issaturated_4_only] = 4

    sources, nsource = label(saturated)
    print('nsources=', nsource, flush=True)
    sizes = sum_labels(saturated, sources, np.arange(nsource)+1)
    msfe = min_sep_from_edge

    # which sources are edge sources?  Anything w/ more than edge_npix contiguous "saturated" pixels
    # add +1 because 0 is the non-saturated zone that we've excluded
    edge_ids = np.where(sizes > edge_npix)[0] + 1
    edge_mask = np.isin(sources, edge_ids)
    saturated = saturated & (~ndimage.binary_dilation(edge_mask, iterations=msfe))


    big_grid = get_psf(header, path_prefix=path_prefix)
    ww = wcs.WCS(header)

    # We force the centroid to be fixed b/c the fitter doesn't do a great job with this...
    # ....this is not optimal...
    #big_grid.fixed['x_0'] = True
    #big_grid.fixed['y_0'] = True

    # daogroup should be set super high to avoid fitting lots of "stars"... if there are a lot of saturated pixels near each other, they're probably all junk
    daogroup = SourceGrouper(min_separation=25)

    resid = data

    lmfitter = LevMarLSQFitter()
    # def levmarverbosewrapper(self, *args, **kwargs):
    #     print("Running lmfitter")
    #     log.info(f"Running lmfitter with args {args} and kwargs {kwargs}")
    #     return self(*args, **kwargs)
    # #lmfitter.__call__ = levmarverbosewrapper
    # lmfitter._run_fitter = levmarverbosewrapper

    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')

    
    slices = find_objects(saturated)
    
    # define pad/size/fwhm_pix (choose sensible defaults if not already set)
    size = pad = 81


    print(f"Found {nsource} saturated sources to process", flush=True)
    index=0
    for ii in range(nsource):
        if True: # to keep indentation level same
            # get the center of pixels with this label
            
            com = center_of_mass(saturated, labels=sources, index=ii+1)
            # center_of_mass can return (nan, nan) for degenerate labels; guard against that
            if com is None:
                print(f"Source {ii+1}: center_of_mass returned None; skipping", flush=True)
                continue
            yf, xf = com
            if not (np.isfinite(yf) and np.isfinite(xf)):
                print(f"Source {ii+1}: center_of_mass returned NaN or infinite values ({yf}, {xf}); skipping", flush=True)
                continue
            ycen = int(round(yf))
            xcen = int(round(xf))
            print(f"Source {ii+1}: center at (x, y) = ({xcen}, {ycen})")
            y0 = int(max(0, ycen - pad))
            y1 = int(min(data.shape[0], ycen + pad))
            x0 = int(max(0, xcen - pad))
            x1 = int(min(data.shape[1], xcen + pad))
            size_saturated = int(np.sqrt(sum_labels(saturated, labels=sources, index=ii+1))/2)
            area_saturated = sum_labels(saturated, labels=sources, index=ii+1)
            cutout = data[y0:y1, x0:x1]
            init_params = QTable()
            init_params['x'] = [xcen - x0]
            init_params['y'] = [ycen - y0]
            cutout[np.isnan(cutout)] = 0.0
    

        
            psfphot = PSFPhotometry(
                                        localbkg_estimator=LocalBackground(15, 30),
                                        fitter=lmfitter,
                                        psf_model=big_grid,
                                        fit_shape=size,
                                        aperture_radius=15*fwhm_pix)
            low_x  = xcen - x0 - size_saturated
            high_x = xcen - x0 + size_saturated
            low_y  = ycen - y0 - size_saturated
            high_y = ycen - y0 + size_saturated

            # get the underlying model and set bounds there
            model = getattr(psfphot, "psf_model", None)
            if model is None:
                raise RuntimeError("psfphot.psf_model is None — can't set parameter bounds")

            for pname, bounds in (("x_0", (low_x, high_x)), ("y_0", (low_y, high_y))):
                if hasattr(model, pname):
                    param = getattr(model, pname)
                    # try the supported API first 
                    try:
                        param.bounds = bounds
                        print(f"Set {pname}.bounds = {bounds}")
                    except (AttributeError, TypeError) as e:
                        # fallback (private attribute) if necessary
                        try:
                            param._bounds = bounds
                            print(f"Set {pname}._bounds = {bounds} (fallback)")
                        except (AttributeError, TypeError) as e:
                            print(f"Could not set bounds for {pname}; parameter object: {param}. Error: {e}")
                else:
                    print(f"Model does not have parameter '{pname}'; param_names={getattr(model,'param_names',None)}")
            # let x_0 be bounds at saturated pixels
            #psfphot.x_0.bounds = (xcen - x0 - size_saturated , xcen - x0 + size_saturated)
            #psfphot.y_0.bounds = (ycen - y0 - size_saturated, ycen - y0 + size_saturated)
            saturated_mask = saturated[y0:y1, x0:x1]
            #pixels_around_saturated = ndimage.binary_dilation(saturated_mask, iterations=5)
            #contrast = np.nanstd(cutout[pixels_around_saturated]) 
            


            # expand a few pixels of saturated area to be masked
            saturated_mask_expanded = ndimage.binary_dilation(saturated_mask, iterations=mask_buffer)
            mask = np.logical_or(cutout==0, np.isnan(cutout), saturated_mask_expanded)
            try:
                result = psfphot(cutout, init_params=init_params, mask=mask)
            except Exception as ex:
                print(f"PSF photometry failed for source {ii+1} at (x,y)=({xcen},{ycen}): {ex}", flush=True)
                continue
            
            result['xcentroid'] = result['x_fit'] + x0
            result['ycentroid'] = result['y_fit'] + y0
            #result['skycoord_fit'] = ww.pixel_to_world(result['xcentroid'], result['ycentroid'])

            
            #result.pprint(max_width=-1)

            ny = cutout.shape[0]
            nx = cutout.shape[1]
            model_image = np.zeros_like(cutout)
            model_err = np.zeros_like(cutout)


            for x_fit, y_fit, flux in zip(result['x_fit'], result['y_fit'], result['flux_fit']):
                # Make a local grid around the source
                if np.isnan(flux):
                    raise ValueError("Flux is NaN; cannot build PSF model image")
                y, x = np.mgrid[0:ny, 0:nx]
                #psf_eval = big_grid(x, y, flux=flux, x_0=x0, y_0=y0)  # works for analytic PSF
                psf_eval = big_grid(x-x_fit, y-y_fit) * flux  # works for GriddedPSFModel
                # cut psf_eval to the image size
                model_image += psf_eval[0:ny, 0:nx]
                model_err += (psf_eval[0:ny, 0:nx] / flux * result['flux_err'][0])**2  # psf_eval already carries flux

            # --- sidelobe residual-flux diagnostic (informational only -- see below) ---
            # If the connected "saturated" region actually merges the real star
            # with a second, unrelated star (or with extended/diffuse emission),
            # a single-PSF fit can only reproduce the compact core it was centered
            # on, and the data just outside the saturated mask carries flux the
            # model has no way to produce. In principle that shows up as a large
            # excess in the residual (cutout - model) there. Because a genuinely
            # merged blob can be much larger than a single PSF's core, the region
            # checked here is sized adaptively from area_saturated rather than a
            # fixed number of pixels.
            #
            # In practice, on real W51 MIRI/NIRCam data this statistic did NOT
            # separate confirmed-bad (merged/extended) sources from confirmed-good
            # (clean, compact) ones -- e.g. on one calibration exposure, known-good
            # sources scored 10-66 while known-bad sources scored 13-21, fully
            # overlapping. The likely reason: W51's diffuse emission is spatially
            # correlated on scales larger than a pixel, so summing residual flux
            # over many annulus pixels and normalizing by sqrt(n_pixels) -- which
            # assumes independent per-pixel noise -- systematically overstates
            # significance regardless of whether the fit is actually bad. Getting
            # this right would need a noise model that accounts for that spatial
            # correlation (or a different formulation entirely), so this is kept
            # as a stored, non-filtering diagnostic column only. `area_ratio`
            # (below) is the check that actually catches merged/extended sources
            # in this dataset.
            residual = cutout - model_image
            r_saturated = np.sqrt(area_saturated / np.pi)  # effective radius of the saturated blob
            outer_iterations = max(int(round(2 * r_saturated)), int(round(3 * fwhm_pix)), mask_buffer + 1)
            outer_mask = ndimage.binary_dilation(saturated_mask_expanded, iterations=outer_iterations)
            annulus = outer_mask & ~saturated_mask_expanded & ~mask

            sidelobe_residual_snr = np.nan
            n_annulus_pix = int(np.sum(annulus))
            if n_annulus_pix >= 8:
                # robust *local* background level/scatter from this same annulus
                _, ann_median, ann_std = sigma_clipped_stats(cutout[annulus])
                if ann_std > 0:
                    excess = residual[annulus] - ann_median
                    sidelobe_residual_snr = np.sum(excess) / (ann_std * np.sqrt(n_annulus_pix))
            """
            threshold_image = np.zeros_like(cutout) 

            #count the number of pixels above local background in the model_image
            threshold = np.nanpercentile(cutout, 99)
            threshold_image[model_image>threshold]=1
            num_pixels_above_threshold = np.nansum(threshold_image)
            print(np.nanmax(model_image), flush=True)
            print(f"Number of pixels above threshold ({threshold}): {num_pixels_above_threshold}", flush=True)
            """
            if len(result) > 0:
                flux = result['flux_fit'][0]
                fluxerr = result['flux_err'][0]
                snr = result['flux_fit'][0] / result['flux_err'][0]
                qfit = result['qfit'][0]
                cfit = result['cfit'][0]
                

            
            
            # check whether the pixels with dqflag = saturated are also flagged as HOT, DEAD, and RC
            # this is a sanity check to make sure that the saturated pixels are not being used in the fit
            idx_saturated_in_cutout = (dq[y0:y1, x0:x1] & dqflags.pixel['SATURATED']) > 0
            # restrict to pixels belonging to the source actually being fit; otherwise
            # saturated pixels from an unrelated star sharing this cutout would get
            # labeled with this source's PSF-fit flux
            idx_saturated_in_cutout &= (sources[y0:y1, x0:x1] == ii + 1)
            # HOT pixels: leave them out of the labels and keep the rest of the source
            idx_saturated_in_cutout &= (dq[y0:y1, x0:x1] & dqflags.pixel['HOT']) == 0
            if not idx_saturated_in_cutout.any():
                print(f"Warning: all saturated pixels are HOT; skipping source", flush=True)
                continue
            saturated_dqflags = dq[y0:y1, x0:x1][idx_saturated_in_cutout]
            if np.any((saturated_dqflags & dqflags.pixel['DEAD'])!=0):
                print(f"Warning: Some saturated pixels are flagged as DEAD; skipping source", flush=True)
                continue



            
            # convert idx_saturated_in_cutout to the index from the original image
            npwhere_2d_idx_saturated_in_cutout = np.where(idx_saturated_in_cutout)
            npwhere_2d_idx_saturated_in_original = (npwhere_2d_idx_saturated_in_cutout[0] + y0, npwhere_2d_idx_saturated_in_cutout[1] + x0)
            # flatten the 2d index to 1d index
            npwhere_1d_idx_saturated_in_original = np.ravel_multi_index(npwhere_2d_idx_saturated_in_original, dims=dq.shape)
            saturated_dqflags = dq[y0:y1, x0:x1][idx_saturated_in_cutout]
            if np.any((saturated_dqflags & dqflags.pixel['DEAD'])!=0):
                print(f"Warning: Some saturated pixels are flagged as DEAD; skipping source", flush=True)
                continue
            #if np.any((saturated_dqflags & dqflags.pixel['RC'])!=0):
            #    print(f"Warning: Some saturated pixels are flagged as RC; skipping source", flush=True)
            #    continue
            threshold_image = np.zeros_like(cutout)

            #count the number of pixels above local background in the model_image
            threshold = np.nanpercentile(cutout, 99)
            threshold_image[model_image>threshold]=1
            num_pixels_above_threshold = np.nansum(threshold_image)
            print(np.nanmax(model_image), flush=True)
            print(f"Number of pixels above threshold ({threshold}): {num_pixels_above_threshold}", flush=True)
           
            if num_pixels_above_threshold< 5:
                print(f"Warning: Only {num_pixels_above_threshold} pixels above threshold; skipping source", flush=True)
                continue

            # compare the observed saturated footprint to the footprint the fitted
            # point-source model itself would produce above the same threshold. A
            # clean, isolated saturated star has area_saturated ~ num_pixels_above_threshold;
            # a saturated region that merges a star with contiguous extended/diffuse
            # emission (or another blended source) has a footprint far larger than
            # any point source could produce, and area_ratio blows up. Empirically
            # (see calibration on real MIRI F1000W sources), clean cores give
            # ratio ~0.2-2, while confirmed merged/extended cases give ratio ~10-14;
            # the default threshold is a first cut from that small sample and should
            # be revisited once run across more data.
            area_ratio = area_saturated / max(num_pixels_above_threshold, 1)
            if area_ratio > area_ratio_threshold:
                print(f"Warning: saturated area ({area_saturated} px) is {area_ratio:.1f}x larger than the "
                      f"model's own footprint ({num_pixels_above_threshold:.0f} px) at threshold={area_ratio_threshold}; "
                      f"likely merged/extended saturation, skipping source", flush=True)
                continue

            # process the result
            if result is not None and np.isfinite(fluxerr) and snr > 1 and flux > 0:
                if index == 0:
                    base_tab = result
                else:
                    base_tab = table.vstack([base_tab, result])
                
                index += 1


                # replace the saturated pixels in the original images with model values.
                cutout_copied = cutout.copy()
                cutout_copied[idx_saturated_in_cutout] = model_image[idx_saturated_in_cutout]
                
                # check the difference made in unsaturated pixels are greater than 20%

                diff = (np.sum(cutout_copied[~idx_saturated_in_cutout]) - np.sum(cutout[~idx_saturated_in_cutout]))/np.sum(cutout[~idx_saturated_in_cutout])
                if np.abs(diff) > 0.2:
                    fig = plt.figure(figsize=(18, 6))
                    ax1 = fig.add_subplot(1,3,1)
                    ax1.imshow(cutout, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
                    ax1.set_title('Cutout')
                    ax2 = fig.add_subplot(1,3,2)
                    ax2.set_title('Model')
                    ax2.imshow(model_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
                    ax3 = fig.add_subplot(1,3,3)
                    ax3.imshow(cutout_copied, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
                    ax3.set_title('Cutout with Saturated Pixels Replaced')
                    # print flux, fluxerr, snr, cfit in the title
                    

                    plt.show()
                    plt.close()
                    print(f"Warning: Replacing saturated pixels changed the total flux in unsaturated pixels by {diff*100:.1f}%; skipping source", flush=True)
                    continue
                


                if plot:
                    fig = plt.figure(figsize=(12, 12))
                    ax1 = fig.add_subplot(2,2,1)
                    ax1.imshow(cutout, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
                    ax1.set_title('Cutout')
                    ax2 = fig.add_subplot(2,2,2)
                    ax2.set_title('Model')
                    ax2.imshow(model_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
                    ax3 = fig.add_subplot(2,2,3)
                    ax3.imshow(cutout-model_image, origin='lower', cmap='viridis', vmin=0, vmax=np.nanpercentile(cutout, 99))
                    ax3.set_title('residual')
                    ax4 = fig.add_subplot(2,2,4)
                    #mark saturated pixel
                    empty_img = np.zeros_like(cutout_copied)
                    empty_img[idx_saturated_in_cutout] = 1
                    ax4.imshow(empty_img, origin='lower', cmap='viridis', vmin=0, vmax=1)
                    ax4.set_title('Saturated Pixels')
                    # print flux, fluxerr, snr, cfit in the title
                    #ax3.set_title(f'Cutout with Saturated Pixels Replaced\nFlux: {flux_vals[0]:.2f}, FluxErr: {fluxerr_vals[0]:.2f}, SNR: {snr:.2f}, Cfit: {cfit:.2f}')
                    
                    img_file_name = fn.replace("_cal.fits", f"_ml_training_{ii}.png")
                    img_file_name = filtername.lower()+img_file_name.split("/")[-1]
                    plt.savefig(f"{img_save_dir}/{img_file_name}")
                    plt.close()
                num_sat_pixels = len(npwhere_1d_idx_saturated_in_original)
                # get 2D coords in the original image (row, col)
                rows = npwhere_2d_idx_saturated_in_original[0]
                cols = npwhere_2d_idx_saturated_in_original[1]

                # flux and fluxerr (length N)
                # the ramp records source + background; PSFPhotometry subtracted local_bkg before fitting
                local_bkg = result['local_bkg'][0]
                flux_vals = model_image[idx_saturated_in_cutout].ravel() + local_bkg
                # model_err accumulates squared terms (variance); take sqrt to get
                # a standard-deviation-like error on the same scale as flux
                fluxerr_vals = np.sqrt(model_err[idx_saturated_in_cutout]).ravel()
                if len(results_sats) == 0:
                    results_sats['x'] = cols
                    results_sats['y'] = rows
                    results_sats['flux'] = flux_vals
                    results_sats['fluxerr'] = fluxerr_vals
                    results_sats['sat_level'] = sat_level_map[rows, cols].ravel()
                    results_sats['DN0'] = rampdata['SCI'].data[0,0, rows, cols]
                    results_sats['DN1'] = rampdata['SCI'].data[0,1, rows, cols]
                    results_sats['DN2'] = rampdata['SCI'].data[0,2, rows, cols]
                    results_sats['DN3'] = rampdata['SCI'].data[0,3, rows, cols]
                    results_sats['DN4'] = rampdata['SCI'].data[0,4, rows, cols]
                    results_sats['slope01'] = results_sats['DN1'] - results_sats['DN0']
                    results_sats['slope12'] = results_sats['DN2'] - results_sats['DN1']
                    results_sats['slope23'] = results_sats['DN3'] - results_sats['DN2']
                    results_sats['slope34'] = results_sats['DN4'] - results_sats['DN3']
                    results_sats['slope_early'] = (results_sats['DN2']-results_sats['DN0'])/2
                    results_sats['curv0'] = results_sats['slope12'] - results_sats['slope01']
                    results_sats['curv1'] = results_sats['slope23'] - results_sats['slope12']
                    results_sats['curv2'] = results_sats['slope34'] - results_sats['slope23']
                    results_sats['nl_strength'] = 1 - np.min([results_sats['curv0'], results_sats['curv1'], results_sats['curv2']], axis=0) 
                    results_sats['max_DN'] = np.max([results_sats['DN1'], results_sats['DN2'], results_sats['DN3'], results_sats['DN4']], axis=0)
                    results_sats['mean_DN'] = np.mean([results_sats['DN1'], results_sats['DN2'], results_sats['DN3'], results_sats['DN4']], axis=0)
                    results_sats['wavelength'] = np.ones(num_sat_pixels) * int(filtername[1:-1])
                    results_sats['bandwidth'] = np.ones(num_sat_pixels) *  filter_bandwidth
                    if instrument == 'NIRCam':
                        results_sats['instrument'] = np.zeros(num_sat_pixels)
                    else:
                        results_sats['instrument'] = np.ones(num_sat_pixels)
                    results_sats['detector'] = np.zeros(num_sat_pixels)*DETECTOR_ID[detector.upper()]
                    
                    results_sats['dist_to_peak'] = np.sqrt((result['x_fit'] + x0 - cols)**2 + (result['y_fit'] + y0 - rows)**2)
                    results_sats['is_near_peak'] = results_sats['dist_to_peak'] < 5  # within 3 pixels of a known star
                    results_sats['psf_star_flux'] = flux
                    results_sats['local_bkg'] = np.ones(num_sat_pixels) * local_bkg
                    results_sats['cal_value'] = cal_value_map[rows, cols]
                    results_sats['fqpm_quadrant'] = fqpm_quadrant_map[rows, cols]
                    results_sats['in_4qpm'] = fqpm_quadrant_map[rows, cols] > 0
                    results_sats['sidelobe_residual_snr'] = np.ones(num_sat_pixels) * sidelobe_residual_snr
                    results_sats['area_ratio'] = np.ones(num_sat_pixels) * area_ratio
                else:
                    new_tab = Table()
                    new_tab['x'] = cols 
                    new_tab['y'] = rows
                    new_tab['flux'] = flux_vals
                    new_tab['fluxerr'] = fluxerr_vals
                    new_tab['sat_level'] = sat_level_map[rows, cols].ravel()
                    new_tab['DN0'] = rampdata['SCI'].data[0,0, rows, cols]
                    new_tab['DN1'] = rampdata['SCI'].data[0,1, rows, cols]
                    new_tab['DN2'] = rampdata['SCI'].data[0,2, rows, cols]
                    new_tab['DN3'] = rampdata['SCI'].data[0,3, rows, cols]
                    new_tab['DN4'] = rampdata['SCI'].data[0,4, rows, cols]
                    new_tab['slope01'] = new_tab['DN1'] - new_tab['DN0']
                    new_tab['slope12'] = new_tab['DN2'] - new_tab['DN1']
                    new_tab['slope23'] = new_tab['DN3'] - new_tab['DN2']
                    new_tab['slope34'] = new_tab['DN4'] - new_tab['DN3']
                    new_tab['slope_early'] = (new_tab['DN2']-new_tab['DN0'])/2
                    new_tab['curv0'] = new_tab['slope12'] - new_tab['slope01']
                    new_tab['curv1'] = new_tab['slope23'] - new_tab['slope12']
                    new_tab['curv2'] = new_tab['slope34'] - new_tab['slope23']
                    new_tab['nl_strength'] = 1 - np.min([new_tab['curv0'], new_tab['curv1'], new_tab['curv2']], axis=0) 
                    new_tab['max_DN'] = np.max([new_tab['DN1'], new_tab['DN2'], new_tab['DN3'], new_tab['DN4']], axis=0)
                    new_tab['mean_DN'] = np.mean([new_tab['DN1'], new_tab['DN2'], new_tab['DN3'], new_tab['DN4']], axis=0)
                    new_tab['wavelength'] = np.ones(num_sat_pixels) * int(filtername[1:-1])
                    new_tab['bandwidth'] = np.ones(num_sat_pixels) *  filter_bandwidth 
                    if instrument == 'NIRCam':
                        new_tab['instrument'] = np.zeros(num_sat_pixels)
                    else:
                        new_tab['instrument'] = np.ones(num_sat_pixels)
                    new_tab['detector'] = np.zeros(num_sat_pixels)*DETECTOR_ID[detector.upper()]
                    
                    new_tab['dist_to_peak'] = np.sqrt((result['x_fit'] + x0 - cols)**2 + (result['y_fit'] + y0 - rows)**2)
                    new_tab['is_near_peak'] = new_tab['dist_to_peak'] < 5  # within 3 pixels of a known star
                    new_tab['psf_star_flux'] = flux
                    new_tab['local_bkg'] = np.ones(num_sat_pixels) * local_bkg
                    new_tab['cal_value'] = cal_value_map[rows, cols]
                    new_tab['fqpm_quadrant'] = fqpm_quadrant_map[rows, cols]
                    new_tab['in_4qpm'] = fqpm_quadrant_map[rows, cols] > 0
                    new_tab['sidelobe_residual_snr'] = np.ones(num_sat_pixels) * sidelobe_residual_snr
                    new_tab['area_ratio'] = np.ones(num_sat_pixels) * area_ratio
                    results_sats = table.vstack([results_sats, new_tab])




            else:
                print(f"Skipping source {ii+1} due to non-finite flux error or low SNR", flush=True)
                print(f"  fluxerr={fluxerr}, snr={snr}", flush=True)
    return results_sats
    """
    # classfy which level each pixel is saturated using groupdq
    issaturated_0 = (groupdq[0,0,:,:] & dqflags.group['SATURATED'])>0
    issaturated_1 = (groupdq[0,1,:,:] & dqflags.group['SATURATED'])>0
    issaturated_2 = (groupdq[0,2,:,:] & dqflags.group['SATURATED'])>0
    issaturated_3 = (groupdq[0,3,:,:] & dqflags.group['SATURATED'])>0
    issaturated_4 = (groupdq[0,4,:,:] & dqflags.group['SATURATED'])>0
    
    issaturated_4_only = issaturated_4 & ~issaturated_3 & ~issaturated_2 & ~issaturated_1 & ~issaturated_0
    issaturated_3_only = issaturated_3 & ~issaturated_2 & ~issaturated_1 & ~issaturated_0
    issaturated_2_only = issaturated_2 & ~issaturated_1 & ~issaturated_0
    issaturated_1_only = issaturated_1 & ~issaturated_0
    issaturated_0_only = issaturated_0

    # print the number of pixels in each category
    print(f"Number of pixels saturated in group 0 only: {np.sum(issaturated_0_only)}", flush=True)
    print(f"Number of pixels saturated in group 1 only: {np.sum(issaturated_1_only)}", flush=True)
    print(f"Number of pixels saturated in group 2 only: {np.sum(issaturated_2_only)}", flush=True)
    print(f"Number of pixels saturated in group 3 only: {np.sum(issaturated_3_only)}", flush=True)
    print(f"Number of pixels saturated in group 4 only: {np.sum(issaturated_4_only)}", flush=True)

    # sample N pixels in each category, where N is the minimum number of pixels in any category (to balance the classes)
    N = min(np.sum(issaturated_0_only), np.sum(issaturated_1_only), np.sum(issaturated_2_only), np.sum(issaturated_3_only), np.sum(issaturated_4_only))
    print(f"Sampling {N} pixels from each category to balance classes", flush=True)
    sampled_0 = np.random.choice(np.where(issaturated_0_only)[0], size=N, replace=False)
    sampled_1 = np.random.choice(np.where(issaturated_1_only)[0], size=N, replace=False)
    sampled_2 = np.random.choice(np.where(issaturated_2_only)[0], size=N, replace=False)
    sampled_3 = np.random.choice(np.where(issaturated_3_only)[0], size=N, replace=False)
    sampled_4 = np.random.choice(np.where(issaturated_4_only)[0], size=N, replace=False)



            
    # if base_tab is not defined, return None
    # this happens if no saturated stars are found
    if index == 0:
        print('No saturated stars found after processing all sources', flush=True)
        return None
    else:
        return base_tab
    """
DETECTOR_ID = {
    "NRCA1": 0,
    "NRCA2": 1,
    "NRCA3": 2,
    "NRCA4": 3,
    "NRCA5": 4,
    "NRCB1": 5,
    "NRCB2": 6,
    "NRCB3": 7,
    "NRCB4": 8,
    "NRCB5": 9,
    "NRCALONG": 20,
    "NRCBLONG": 21,
    "MIRIMAGE": 40,
}

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                    default='F140M',
                    help="filter name list", metavar="filternames")
    parser.add_option("--proposal_id", dest="proposal_id",
                    default='6151',
                    help="proposal_id", metavar="proposal_id")
    parser.add_option("--target", dest="target",
                    default='w51',
                    help="target", metavar="target")
    (options, args) = parser.parse_args()

    filt = options.filternames.split(",")
    filts = [f.strip().upper() for f in options.filternames.split(",") if f.strip()]

    proposal_id = options.proposal_id
    target = options.target

    nvisits = {'2221': {'brick': 1, 'cloudc': 2},
                '1182': {'brick': 2},
                '6151': {'w51': 1, 'w51_miri': 2}
                }
    field_to_reg_mapping = {'2221': {'001': 'brick', '002': 'cloudc'},
                            '1182': {'004': 'brick'},
                            '6151': {'001': 'w51', '002':'w51_miri'}}[proposal_id]
    reg_to_field_mapping = {v:k for k,v in field_to_reg_mapping.items()}
    field = reg_to_field_mapping[target]

    basepath = f'/orange/adamginsburg/jwst/w51'
    out_dir = '/orange/adamginsburg/w51/TaehwaYoo/w51_sat_recover/v3'
    os.makedirs(f'{out_dir}/cutout_img', exist_ok=True)
    

    nircam_short_filters = ['F140M', 'F162M', 'F182M', 'F187N', 'F200W', 'F210M', 'F277W', ]
    nircam_long_filters = ['F300M', 'F335M', 'F356W', 'F360M','F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M']
    miri_filters = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
    
    for filt in filts:
       
        if filt in ['F140M', 'F162M', 'F182M', 'F187N', 'F210M']:
            modules = ('nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4')
        elif filt in ['F335M', 'F360M', 'F405N', 'F410M', 'F480M']:
            modules = ('nrcalong', 'nrcblong')
        else:
            modules = ('mirim',)
        print(modules)
        for module in modules:
            if filt in ['F140M', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M', 'F405N', 'F410M', 'F480M']:
                globlist = glob.glob(f"/orange/adamginsburg/jwst/w51/{filt}/pipeline/*{module}*_cal.fits")
              
                instrument = 'NIRCam'
            else:
                # MIRI: jwst_new_cal_claude Detector1 settings with the saturation step on
                globlist = glob.glob(f"/orange/adamginsburg/w51/jwst/jwst_new_cal_claude_satstep/{filt}/pipeline_4qpm/*mirimage_cal.fits")
                instrument = 'MIRI'
            print("filt=", filt, "modules=", modules, flush=True)
            print("n files=", len(globlist), flush=True)
            for ii, fn in enumerate(globlist):
                print()
                if ii==0:
                    plot=True
                else:
                    plot=False
                bn = os.path.basename(fn)
                exposurenumber = int(bn.split("_")[2])
                exposure_id = bn.split("_")[2]
                visit_id = bn.split("_")[0][-3:]
                vgroup_id = bn.split("_")[1]
                exposure_ = f'_exp{exposurenumber:05d}'
                visitid_ = f'_visit{int(visit_id):03d}' if visit_id is not None else ''
                if instrument == 'NIRCam':
                    vgroupid_ = f'_vgroup{int(vgroup_id):05d}' if vgroup_id is not None else ''
                elif instrument == 'MIRI':
                    vgroupid_ = f'_vgroup{vgroup_id}' if vgroup_id is not None else ''
                    module2 = 'mirimage'
                detector = bn.split("_")[-2]
                print(detector, flush=True)
                #f360m_nrcb_visit001_vgroup3105_exp00008_daophot_basic.fits
                #/orange/adamginsburg/jwst/w51/F360M/f360m_nrcalong_visit001_vgroup3105_exp00005_daophot_basic.fits
                print('vgroup_id', vgroup_id, flush=True)
                print('vgroupid_', vgroupid_, flush=True)
                wav = int(filt[1:4])
                #fitsdata, rampdata
                cal_fits = fits.open(fn)
                if instrument == 'MIRI':
                    # 4QPM panes are NaN in _cal (no imager flat); _bkgm has them calibrated, shifted by the
                    # exposure's background-match offset, which is removed to stay on the _cal scale
                    with fits.open(fn.replace('_cal.fits', '_bkgm.fits')) as bkgm:
                        sci, var = cal_fits['SCI'].data, cal_fits['VAR_POISSON'].data
                        offset = np.nanmedian(bkgm['SCI'].data[:, miri_4qpm.IMAGER_X0:] - sci[:, miri_4qpm.IMAGER_X0:])
                        pane = miri_4qpm.pane_quadrant_image(line_halfwidth=0) > 0
                        sci[pane] = bkgm['SCI'].data[pane] - offset
                        var[pane] = bkgm['VAR_POISSON'].data[pane]
                ramp_file = fn.replace("_cal.fits", "_ramp.fits")
                ramp_fits = fits.open(ramp_file)
                #ramp_fits = RampModel(ramp_file)
                if True:
                   
                    training_data = collect_saturated_pixels(cal_fits, ramp_fits, filtername=filt, instrument=instrument, detector=detector, plot=plot, fn=fn, img_save_dir=f'{out_dir}/cutout_img')
                    #save the training data to a fits file
                    save_file_name = fn.replace("_cal.fits", "_ml_training.fits")
                    save_file_name = save_file_name.split("/")[-1]
                    save_file_name = f"{filt.lower()}_"+save_file_name
                    training_data.write(f"{out_dir}/{save_file_name}", overwrite=True)

            #for module in ('nrca', 'nrcb', 'merged'):
            #    for fn in glob.glob(f"/orange/adamginsburg/jwst/w51/F*/pipeline/*{module}*crf.fits"):
            #        print()
            #        print(fn)
        #        remove_saturated_stars(fn, verbose=True)
if __name__ == "__main__":
    main()

