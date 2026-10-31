import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import QTable, Table
from scipy.interpolate import interp1d
import functools
import logging
import h5py

from simple.tools_python import (
    print_memory_usage,
    transform_bin_to_h5,
    bin_scipy,
    jinc,
)
from simple.simple import make_map, LognormalIntensityMock
from simple.tools import get_kspec_cython


def kaiser_pkmu(Plin, k, mu, bias, f_growth):
    r"""
    Returns the Kaiser linear P(k,mu) in redshift space

    .. math::

        P(k,mu) = (1 + f/b_1 \mu^2)^2 b_1^2 P_\mathrm{lin}(k)

    Copied from nbodykit.

    """
    return (bias + f_growth(k) * mu**2) ** 2 * Plin(k)


class Power_Spectrum_Model(LognormalIntensityMock):
    """
    Power_Spectrum_Model class. 
    Models the intensity/galaxy/cross power spectrum of a LognormalIntensityMock instance
    with the same input parameters as that.

    Parameters
    ----------
    input_dict:
        Input parameters for the LognormalIntensityMock instance, see simple.py.
    do_model_shot_noise: bool
        Set to True to model the shot noise by shuffling the galaxy positions perpendicular to the LOS.
        Default is None/False.
    out_filename: str
        Name of the file to save the model power spectra to (hdf5 format).

    Notes
    -----
    The general model is explained in the paper.

    """

    def __init__(
        self, 
        input_dict, 
        do_model_shot_noise = None, 
        out_filename = None,
        galaxy_mask = None,
        intensity_mask = None,
    ):
        LognormalIntensityMock.__init__(self, input_dict=input_dict)
        self.own_init(do_model_shot_noise, out_filename, galaxy_mask, intensity_mask)

    @classmethod
    def from_file(
        cls,
        filename,
        out_filename,
        catalog_filename=None,
        only_params=False,
        only_meshes=["noise_mesh", "obs_mask"],
        do_model_shot_noise=None,
        galaxy_mask = None,
        intensity_mask = None,
    ):
        """
        Method to initiate a Power_Spectrum_Model object from a file
        containing a saved LognormalIntensityMock object.

        Parameters
        -----------
        filename : str
            Name of the file containing the saved LognormalIntensityMock object.
        out_filename : str
            Output filename for the power spectrum results.
        catalog_filename : str, optional
            Filename of the catalog associated with the LognormalIntensityMock object.
        only_params : bool, optional
            Set to True to load only the parameters and skip loading the meshes.
        only_meshes : list of str, optional
            List of specific meshes to load. Default is ["noise_mesh", "obs_mask"].
        do_model_shot_noise : bool, optional
            Set to True to model the shot noise by shuffling the galaxy positions perpendicular to the LOS.
            Default is None/False.

        Returns
        --------
        instance : Power_Spectrum_Model
            An instance of the Power_Spectrum_Model class.

        """

        instance = super().from_file(filename=filename,
                                     catalog_filename=catalog_filename,
                                     only_params=only_params,
                                     only_meshes=only_meshes,
                                     )
        instance.own_init(do_model_shot_noise, out_filename, galaxy_mask, intensity_mask)
        return instance

    def own_init(self, do_model_shot_noise, out_filename, galaxy_mask, intensity_mask):
        """
        Step in initializing the Power_Spectrum_Model instance.

        Parameters
        -----------
        do_model_shot_noise : bool
            Set to True to model the shot noise by shuffling the galaxy positions perpendicular to the LOS.
        out_filename : str
            Output filename for the power spectrum results.

        """

        logging.info("Initializing Power_Spectrum_Model instance.")
        self.do_model_shot_noise = do_model_shot_noise
        self.out_filename = out_filename

        self.galaxy_mask = galaxy_mask
        self.intensity_mask = intensity_mask

        if isinstance(self.input_pk_filename, str):
            plin_tab = Table.read(self.input_pk_filename, format="ascii")
            self.Plin = interp1d(
                plin_tab["col1"], plin_tab["col2"], fill_value="extrapolate"
            )
        else:
            logging.Error(
                "input_pk_filename must be the string filename of a tabulated power spectrum."
            )

        if isinstance(self.f_growth_filename, str):
            fnu_tab = Table.read(self.f_growth_filename, format="ascii")
            self.f_growth = interp1d(
                fnu_tab["col1"], fnu_tab["col2"], fill_value="extrapolate"
            )
        else:
            logging.Error(
                "f_growth must be the string filename of a tabulated power spectrum."
            )

        self.ngal_shot_noise_model = None
        self.im_shot_noise_model = None
        self.cross_shot_noise_model = None

        if self.obs_mask is None:
            self.obs_mask = np.ones(self.N_mesh, dtype=int)
            logging.info("Assigned ones to mask.")
        if self.galaxy_mask is None:
            self.galaxy_mask = np.ones(self.N_mesh, dtype=int)
            logging.info("Assigned ones to galaxy_mask.")
        if self.intensity_mask is None:
            self.intensity_mask = np.ones(self.N_mesh, dtype=int)
            logging.info("Assigned ones to intensity_mask.")

        self.s_par = self.sigma_par().to(self.Mpch).value
        self.s_perp = self.sigma_perp().to(self.Mpch).value
        try:
            self.s_perp_sky = (
                (
                    self.footprint_radius
                    * self.astropy_cosmo.kpc_comoving_per_arcmin(self.redshift)
                )
                .to(self.Mpch)
                .value
            )
        except Exception as e:
            print(e)
            pass

        logging.info("Done.")

    # get intensity-intensity shot noise
    def luminosity_function_times_Lsq(self, L):
        """ 
        Luminosity function times L**2

        """
        return self.luminosity_function_times_L(L) * L

    def integrate_luminosity_function_times_Lsq(self, L_min, L_max):
        """
        Integral of the luminosity function times L**2 from L_min to L_max
        (in units of self.luminosity_unit), on a log-spaced grid like in
        mean_intensity_per_redshift. scipy's quad misses the luminosity function
        when the range spans many decades (e.g. Lmax = 1e10 * Lmin) and returns 0.

        """
        Ls = np.logspace(np.log10(L_min), np.log10(L_max), 10000)
        return np.trapz(self.luminosity_function_times_Lsq(Ls), Ls)

    def get_observed_volume(self):
        """ Returns the observed volume given the mask."""
        observed_volume = np.sum(self.obs_mask) * self.voxel_volume
        self.observed_volume = observed_volume
        return observed_volume

    def get_S_bar(self, weight_mesh, sigma_noise):
        """ Calculates the S_bar value given the weight mesh and the sigma of the Gaussian noise."""
        return (
            np.mean(weight_mesh**2 * sigma_noise**2).to(1)
            * self.voxel_volume
            / self.box_volume
        )

    def get_Q_bar(self, weight_mesh, mean_intensity_per_redshift_mesh):
        """ Calculates the Q_bar value given the weight mesh and the mean field as a function of position. """
        return np.mean(weight_mesh**2 * mean_intensity_per_redshift_mesh**2).to(1)

    def D_sq_par_of_k(self, k_par, s_par):
        """
        Damping function D^2 for LOS-parallel top-hat smoothing with smoothing length s_par.
        """

        if self.do_spectral_tophat_smooth:
            D_sq_par = np.sinc(0.5 * k_par * s_par / np.pi) ** 2
        else:
            D_sq_par = np.exp(-(k_par**2) * s_par**2)
        return D_sq_par

    def D_sq_perp_of_k(self, k_perp, s_perp):
        """
        Damping function D^2 for LOS-perpendicular exponential smoothing with smoothing length s_perp.
        """

        D_sq_perp = np.exp(-(k_perp**2) * s_perp**2)
        return D_sq_perp

    def D_cic_of_k(self, k, k_N):
        """
        Damping function D for cic resampling with the aliasing correction for the shot noise.
        """
        # D_cic = np.sinc(0.5 * k * H/np.pi)**2 # without aliasing correction
        D_cic = (1 - 2.0 / 3 * np.sin(np.pi * k / (2 * k_N)) ** 2) ** 0.5
        return D_cic

    @functools.cached_property
    def D_cic_corr(self):
        """
        Damping function D for cic resampling evaluated for this instance as a 3D array.
        """

        D_cic_corr = (
            self.D_cic_of_k(self.kx[:, None, None], self.k_N)
            * self.D_cic_of_k(self.ky[None, :, None], self.k_N)
            * self.D_cic_of_k(self.kz[None, None, :], self.k_N)
        )
        return D_cic_corr

    def D_sq_tophat_of_k(self, k_perp, s_perp_sky):
        """
        Damping function D^2 for a spherical top-hat smoothing in the directions perpendicular to the LOS.
        """

        return (jinc(k_perp * s_perp_sky) * 2) ** 2

    def sky_subtraction_factor(self):
        """
        The factor 1 - D_sky(k_perp) on the k grid by which the sky subtraction
        multiplies the map (signal and noise), with D_sky = 2 J_1(k_perp s_f) / (k_perp s_f)
        the top-hat of the sky estimate (Eqs. 22-25 of the SIMPLE paper; s_f = s_perp_sky).
        P gets its square and the cross power the factor itself. D_sky < 0 where
        3.83 < k_perp s_f < 7.02 (and in every second lobe after that), so
        1 - D_sky is not 1 - sqrt(D_sky^2) there.
        """
        return 1 - 2 * jinc(np.asarray(self.k_perp) * self.s_perp_sky)

    @functools.cached_property
    def Pm_kspec(self):
        """
        Cached property that returns the galaxy power spectrum Pm_kspec.
        If self.RSD is True, it will return the Kaiser RSD approximation.
        Otherwise it will be just the bias squared times the matter power spectrum.

        Returns
        --------
        astropy quantity array
            The power spectrum Pm_kspec.

        """

        logging.info("Getting Pm_kspec...")
        if self.RSD:
            Pm_kspec = (
                kaiser_pkmu(
                    self.Plin, self.kspec, self.muspec, self.bias, self.f_growth
                )
                * self.Mpch**3
            )
        else:
            Pm_kspec = self.bias**2 * self.Plin(self.kspec) * self.Mpch**3
        logging.info("Done.")
        return Pm_kspec

    @functools.cached_property
    def k_N(self):
        """ Nyquist frequency in [h/Mpc]."""
        return self.k_Nyquist[0].value

    def removed_modes(self, tracer):
        """
        The modes (True on the k grid) that the estimators remove. The
        intensity estimator subtracts the mean of each slice along the LOS, so
        intensity and cross have no power at k_perp = 0. The galaxy field is
        divided by the expected mean density and only loses k = 0, unless
        min_flux is a mesh: then it loses k_perp = 0 as well.
        """
        if tracer == "n_gal" and not self.min_flux_is_mesh:
            return np.asarray(self.kspec) == 0
        return np.asarray(self.k_perp) == 0

    def convolve_window(self, model, mask_window_function_1, mask_window_function_2=None):
        """
        Convolves a power spectrum on the k grid (array, in (Mpc/h)^3) with the
        window functions of the two fields (Eqs. 11 and 13 of the SIMPLE paper):
        the correlation function is multiplied by the correlation of the two
        windows in real space and transformed back. Returns the convolved
        power on the k grid (array, in (Mpc/h)^3).
        """
        logging.info("Convolving P(k) with the window function.")

        def to_map(field, **kwargs):
            return make_map(field, Nmesh=self.N_mesh,
                            BoxSize=self.box_size.to(self.Mpch).value, **kwargs)

        xi = to_map(model, type="complex").c2r()
        window_ft_1 = to_map(mask_window_function_1).r2c()
        if mask_window_function_2 is None:
            window_ft_2 = window_ft_1
        else:
            window_ft_2 = to_map(mask_window_function_2).r2c()
        window_correlation = (window_ft_1 * np.conjugate(window_ft_2)).c2r()
        return np.array((xi * window_correlation).r2c())

    def white_noise_through_window(self, mask_window_function, noise_power):
        """
        Power on the k grid of a noise that is white before the mask, with the
        power noise_power (array, in (Mpc/h)^3) in each cell, after the mean of
        each slice along the LOS is subtracted (which removes k_perp = 0) and
        the map is multiplied by the window W. With a^2 = noise_power,

            N(k) = mean(W^2 a^2) - mean over the slices of
                   [2 Re(hat(W a^2) conj(hat(W))) - mean_perp(a^2) |hat(W)|^2](k_perp),

        where hat is the 2D Fourier transform of a slice divided by the number
        of cells in the slice, and mean_perp the mean over a slice. The second
        term is there because the slice mean is taken before the mask, over
        all cells of the slice; without a mask it is mean(a^2) at k_perp = 0
        and 0 elsewhere. For noise that is not smoothed, like the pixel noise.
        """
        W = np.broadcast_to(np.asarray(mask_window_function, dtype=float), tuple(self.N_mesh))
        a2 = np.broadcast_to(np.asarray(noise_power, dtype=float), tuple(self.N_mesh))
        n_perp = self.N_mesh[1] * self.N_mesh[2]
        W_hat = np.fft.rfft2(W) / n_perp  # transform of each slice, on the k_perp grid
        Wa2_hat = np.fft.rfft2(W * a2) / n_perp
        a2_perp = np.mean(a2, axis=(1, 2))[:, None, None]
        correction = np.mean(
            2 * np.real(Wa2_hat * np.conj(W_hat)) - a2_perp * np.abs(W_hat)**2, axis=0)
        return np.mean(W**2 * a2) - correction * np.ones(np.shape(self.k_perp))

    def get_intensity_noise(self, sigma_noise, sky_subtraction=False):
        """
        Noise power of the intensity map (divided by its mean) on the k grid,
        through the window, in (Mpc/h)^3. The estimator smooths the map, adds
        the pixel noise, subtracts the mean of each slice (which removes
        k_perp = 0) and multiplies by the mask W. So:

        - the Poisson noise is smoothed like the signal and masked after the
          smoothing. Its power in each slice without the mask is a^2. The
          damping D^2 with k_perp = 0 set to zero is convolved with the window
          W a, as the signal is convolved with W w. This is exact for any W
          (also one that changes along the LOS) if a is the same in all
          slices, or if there is no LOS smoothing. Otherwise it assumes that a
          changes little within the LOS smoothing length, like the signal;
          the error is of second order in s_par * dln(a)/dx_par, also
          without a mask, where the noise is exactly mean(a^2) D^2.
        - the pixel noise is not smoothed: white_noise_through_window with
          a^2 = (sigma_noise / mean intensity)^2 * voxel volume.

        With sky_subtraction, the sky estimate is the map with its noise
        smoothed by the top-hat, so both noises are multiplied by
        sky_subtraction_factor() before the mask, which also removes k_perp = 0.
        The pixel noise is then convolved with the window W a like the Poisson
        noise: exact if a is the same in all cells of each slice (it may change
        along the LOS, since the top-hat only averages within a slice),
        otherwise it assumes that a changes little within s_perp_sky.
        """
        removed = self.removed_modes("intensity")
        W = self.obs_mask * self.intensity_mask
        a_shot = np.sqrt(self.intensity_shot_noise_per_slice().to_value(self.Mpch**3))
        D_sq = self.D_sq_par_of_k(self.k_par, self.s_par) * self.D_sq_perp_of_k(self.k_perp, self.s_perp)
        sky_sq = self.sky_subtraction_factor()**2 if sky_subtraction else 1.0
        shot = self.convolve_window(np.where(removed, 0.0, D_sq * sky_sq), W * a_shot[:, None, None])
        a2_pixel = (
            (sigma_noise * self.weight_mesh_im)**2 * self.voxel_volume).to_value(self.Mpch**3)
        if sky_subtraction:
            pixel = self.convolve_window(np.where(removed, 0.0, sky_sq), W * np.sqrt(a2_pixel))
        else:
            pixel = self.white_noise_through_window(W, a2_pixel)
        return (shot + pixel) * self.Mpch**3

    def get_cross_noise(self, sky_subtraction=False):
        """
        Poisson noise power of the cross power spectrum on the k grid, through
        the windows, in (Mpc/h)^3. The intensity map is smoothed once (damping D,
        the square root of D^2) and loses k_perp = 0 before the mask; the galaxy
        map is not smoothed. c is the cross noise in each slice without the
        mask. As for the intensity (get_intensity_noise), D with k_perp = 0
        set to zero is convolved with the windows W_g and W_I c, which assumes
        that c changes little within the LOS smoothing length. The pixel noise
        does not correlate with the galaxies. With sky_subtraction, D is
        multiplied by sky_subtraction_factor().
        """
        removed = self.removed_modes("cross")
        c = self.cross_shot_noise_per_slice().to_value(self.Mpch**3)
        D = np.sqrt(self.D_sq_par_of_k(self.k_par, self.s_par) * self.D_sq_perp_of_k(self.k_perp, self.s_perp))
        if sky_subtraction:
            D = D * self.sky_subtraction_factor()
        noise = self.convolve_window(
            np.where(removed, 0.0, D),
            self.obs_mask * self.galaxy_mask,
            self.obs_mask * self.intensity_mask * c[:, None, None],
        )
        return noise * self.Mpch**3

    def get_3d_pk_model(
        self,
        damping_function,
        P_shot_smoothed,
        S_bar,
        mask_window_function_1,
        observed_volume,
        box_volume,
        Pk_unit,
        convolve=True,
        mask_window_function_2=None,
        save=True,
        tracer=None,
        return_3d=False,
        noise=None,
    ):
        """
        Calculates the 3D power spectrum model P(k) by incorporating smoothing, convolution with the window function,
        and noise, and bins it in linear k bins.

        Parameters
        -----------
        damping_function : array-like
            The damping function.
        P_shot_smoothed : astropy quantity scalar or array
            The shot noise power spectrum that is subject to smoothing (i.e. not pixel noise) in volume units.
        S_bar : float
            The value of S_bar.
        mask_window_function_1 : array-like
            The window function: mask times mean field as a function of position.
        observed_volume : astropy quantity
            The observed volume in volume units.
        box_volume : astropy quantity
            The total box volume in volume units.
        Pk_unit : astropy unit object
            The unit of the power spectrum (e.g. u.Unit("u.Mpc**3"))
        convolve : bool, optional
            Flag indicating whether to convolve P(k) with the window function squared. 
            If False, just multiply by Q_bar (mean of the window function squared).
            Default is True.
        mask_window_function_2 : array-like, optional
            The window function for the second field in case of a cross power spectrum.
            Default: None.
        save : bool, optional
            Flag indicating whether to save the results. Default is True.
        tracer : str, optional
            Options: n_gal, intensity, cross, sky_subtracted_intensity, sky_subtracted_cross.
            The tracer label in the saved file. Default is None.
        return_3d: bool, optional
            Set to True if you want to return the 3D model as well.
        noise : astropy quantity array, optional
            The noise power on the k grid, already through the window
            (see get_intensity_noise and get_cross_noise).
            If None, S_bar * box_volume + P_shot_smoothed * damping_function
            is added after the window in all but the removed modes, which is
            exact only without a mask.

        Returns:
        --------
        mean_k
            Mean k in the k bins.
        monopole
            Monopole moment of the binned power spectrum.
        quadrupole 
            Quadrupole moment of the binned power spectrum.

        """
        logging.info("Getting 3D P(k) model.")
        print(self.Pm_kspec.unit, P_shot_smoothed.unit)
        model = (self.Pm_kspec) * damping_function  # + P_shot_smoothed
        # Set to zero the modes that the estimators remove, before the window
        # convolution (see removed_modes).
        removed = self.removed_modes(tracer)
        model[removed] = 0
        model = model.to(self.Mpch**3).value
        if convolve:
            model = self.convolve_window(
                model, mask_window_function_1, mask_window_function_2)
        else:
            if mask_window_function_2 is None:
                model = (
                    model
                    * np.mean(mask_window_function_1**2)
                    * observed_volume
                    / box_volume
                )  # Q_bar
            else:
                model = (
                    model
                    * np.mean(mask_window_function_1 * mask_window_function_2)
                    * observed_volume
                    / box_volume
                )  # Q_bar
        print("S_bar * box_volume: ", (S_bar * box_volume))
        print("P_shot_smoothed: {}".format(P_shot_smoothed))
        if noise is None:
            # the estimators remove the noise from these modes too
            noise = (S_bar * box_volume + P_shot_smoothed * damping_function) * ~removed
        model = (np.array(model) * self.Mpch**3 + noise).to(Pk_unit).value

        (
            mean_k,
            monopole,
            quadrupole,
            n_modes
        ) = self.bin_scipy(model.real, return_nmodes=True)

        if save:

            tracer_units = {
                "intensity" : str(self.mean_intensity**2 * self.Mpch**3),
                "cross" : str(self.mean_intensity * self.Mpch**3),
                "n_gal" : str(self.Mpch**3),
                "sky_subtracted_intensity" : str(self.mean_intensity**2 * self.Mpch**3),
                "sky_subtracted_cross" : str(self.mean_intensity * self.Mpch**3),
            }
            with h5py.File(self.out_filename, "a") as ff:
                if tracer in ff.keys():
                    del ff[tracer]
                    logging.info(
                        f'Overwriting {tracer} in file {self.out_filename}.')
                grp = ff.create_group(tracer)
                ff[f"{tracer}/monopole"] = monopole
                ff[f"{tracer}/monopole"].attrs['unit'] = tracer_units[tracer]

                ff[f"{tracer}/quadrupole"] = quadrupole
                ff[f"{tracer}/quadrupole"].attrs['unit'] = tracer_units[tracer]

                ff[f"{tracer}/k_bins"] = mean_k
                ff[f"{tracer}/k_bins"].attrs['unit'] = 1/self.Mpch

                ff[f"{tracer}/P_shot"] = P_shot_smoothed.to(self.Mpch**3).value
                ff[f"{tracer}/P_shot"].attrs['unit'] = tracer_units[tracer]

                ff[f"{tracer}/S_bar"] = (S_bar *
                                         box_volume).to(self.Mpch**3).value
                ff[f"{tracer}/S_bar"].attrs['unit'] = tracer_units[tracer]

                ff[f"{tracer}/n_modes"] = n_modes
                logging.info("Done")
        if return_3d:
            return model, mean_k, monopole, quadrupole
        else:
            return mean_k, monopole, quadrupole

    def model_shot_noise(self, N_real=10):
        """
        Model the shot noise in the intensity and galaxy distributions
        by shuffling the cells in the directions perpendicular to the LOS
        while keeping the LOS position fixed (to keep directional information
        that may exist due to redshift differences).
        After shuffling, calculate the power spectrum.
        Finally the model shot noise is given by the average of the power spectra
        of the shuffled meshes.

        Parameters
        -----------
        N_real : int, optional
            Number of realizations of shuffled meshes. Default is 10.

        """

        N_gal = int((self.n_bar_gal * self.box_volume).to(1))
        intensity_shot_noise_delta_k_sqs = []
        n_gal_shot_noise_delta_k_sqs = []
        cross_shot_noise_delta_k_sqs = []
        # a QTable, so that columns with units are Quantities (with to_value),
        # as in the mock's catalog; apply_selection_function needs that for a min_flux mesh
        catalog = QTable()
        catalog["detected"] = np.ones(N_gal, dtype=bool)
        # catalog['RSD_Position'] = catalog['Position']
        catalog["RSD_redshift_factor"] = np.ones(N_gal, dtype=int)
        catalog["Velocity"] = np.zeros((N_gal, 3), dtype=int) * u.km / u.s
        N_real = N_real
        catalog["Position"] = np.transpose(
            [
                np.random.uniform(
                    low=0.0,
                    high=self.box_size[0].to(self.Mpch).value,
                    size=N_gal,
                ),
                np.random.uniform(
                    low=0.0,
                    high=self.box_size[1].to(self.Mpch).value,
                    size=N_gal,
                ),
                np.random.uniform(
                    low=0.0,
                    high=self.box_size[2].to(self.Mpch).value,
                    size=N_gal,
                ),
            ]
        )
        catalog["Position"] = catalog["Position"] * self.Mpch
        self.cat = catalog
        self.assign_redshift_along_axis()
        self.assign_luminosity()
        self.assign_flux()
        self.apply_selection_function()
        self.do_angular_smooth = False
        self.do_spectral_smooth = False
        self.paint_intensity_mesh(position="Position")
        self.paint_galaxy_mesh(position="Position")
        indices = np.arange(self.N_mesh[1]*self.N_mesh[2])
        intensity_mesh = (self.intensity_mesh).to(
            self.mean_intensity).value
        del self.intensity_mesh
        intensity_mesh = make_map(
            intensity_mesh,
            Nmesh=self.N_mesh,
            BoxSize=self.box_size.to(self.Mpch).value,
        )
        n_gal_mesh = self.n_gal_mesh.to(u.Mpc**(-3)).value
        del self.n_gal_mesh
        n_gal_mesh = make_map(n_gal_mesh,
                              Nmesh=self.N_mesh,
                              BoxSize=self.box_size.to(self.Mpch).value,
                              )
        for i in range(N_real):
            if False:
                self.cat["Position"][:, 1] = np.random.uniform(
                    low=0.0,
                    high=self.box_size[1].to(self.Mpch).value,
                    size=N_gal,
                )
                self.cat["Position"][:, 2] = np.random.uniform(
                    low=0.0,
                    high=self.box_size[2].to(self.Mpch).value,
                    size=N_gal,
                )

            # TODO: shuffle intensity mesh and ngal mesh but keeping the redshift (axis 0) fixed
            intensity_map = np.empty_like(intensity_mesh)
            galaxy_map = np.empty_like(n_gal_mesh)
            for j in range(self.N_mesh[0]):
                np.random.shuffle(indices)
                intensity_map[j] = np.array(np.split(intensity_mesh[j][np.unravel_index(
                    indices, shape=(self.N_mesh[1], self.N_mesh[2]))], self.N_mesh[1]))
                galaxy_map[j] = np.array(np.split(n_gal_mesh[j][np.unravel_index(
                    indices, shape=(self.N_mesh[1], self.N_mesh[2]))], self.N_mesh[1]))

            # get intensity shot noise
            intensity_map = (
                intensity_map - np.mean(intensity_map,
                                        axis=(1, 2))[:, None, None]
            )
            weights_im = 1.  # self.mean_intensity.value  # _per_redshift_mesh
            intensity_rfield = make_map(
                (intensity_map / weights_im),
                Nmesh=self.N_mesh,
                BoxSize=self.box_size.to(self.Mpch).value,
            )
            # intensity_rfield = intensity_rfield.r2c().apply(
            #    self.compensation[0][1], kind=self.compensation[0][2]
            # ).c2r()
            delta_k_im = (intensity_rfield * self.obs_mask * self.intensity_mask).r2c()
            delta_k_sq = delta_k_im * np.conjugate(delta_k_im)
            intensity_shot_noise_delta_k_sqs.append(delta_k_sq)

            # get n_gal shot noise
            mean_ngal_per_z = self.mean_ngal_per_redshift_mesh.to(
                u.Mpc**(-3)).value
            galaxy_map = ((galaxy_map - mean_ngal_per_z) /
                          mean_ngal_per_z)
            galaxy_rfield = make_map(galaxy_map,
                                     Nmesh=self.N_mesh,
                                     BoxSize=self.box_size.to(self.Mpch).value,
                                     )
            # galaxy_rfield = galaxy_rfield.r2c().apply(
            #   self.compensation[0][1], kind=self.compensation[0][2]
            # ).c2r()
            delta_k_ngal = (galaxy_rfield * self.obs_mask * self.galaxy_mask).r2c()
            delta_k_sq = delta_k_ngal * np.conjugate(delta_k_ngal)
            n_gal_shot_noise_delta_k_sqs.append(delta_k_sq)

            # get cross shot noise
            delta_k_sq = delta_k_ngal * np.conjugate(delta_k_im)
            cross_shot_noise_delta_k_sqs.append(delta_k_sq)

            logging.info("Finished {}/{}.".format(i + 1, N_real))

        self.im_shot_noise_model = np.nanmean(intensity_shot_noise_delta_k_sqs, axis=0) * self.box_volume.to(
            self.Mpch**3
        )
        self.ngal_shot_noise_model = np.nanmean(n_gal_shot_noise_delta_k_sqs, axis=0) * self.box_volume.to(
            self.Mpch**3
        )
        self.cross_shot_noise_model = np.nanmean(cross_shot_noise_delta_k_sqs, axis=0) * self.box_volume.to(
            self.Mpch**3
        )
        logging.info("Mean shot noises:")
        logging.info(" - im: {}".format(np.mean(self.im_shot_noise_model)))
        logging.info(" - ngal: {}".format(np.mean(self.ngal_shot_noise_model)))
        logging.info(
            " - cross: {}".format(np.mean(self.cross_shot_noise_model)))
        return

    def intensity_shot_noise_per_slice(self):
        """
        Poisson noise power of the intensity map (divided by its mean) in each
        slice along the LOS, without the mask, in (Mpc/h)^3. It comes from
        randomly sampling the number of galaxies and the luminosity of each
        galaxy (second moment of the luminosity function) and is smoothed
        like the signal.

        """
        selection = self.galaxy_selection["intensity"]
        int_L_min = self.Lmin.to(self.luminosity_unit).value
        int_L_max = self.Lmax.to(self.luminosity_unit).value
        P_shot = []
        for i, redshift in enumerate(self.redshift_mesh_axis):
            if selection in ["detected", "undetected"]:
                if callable(self.min_flux):
                    F_min = self.min_flux(redshift)
                elif self.min_flux_is_mesh:
                    F_min = self.min_flux[i]  # the cells of slice i
                else:
                    F_min = self.min_flux
                int_L_limit = (
                    F_min
                    * 4
                    * np.pi
                    * self.astropy_cosmo.luminosity_distance(redshift) ** 2
                ).to(self.luminosity_unit).value
                if selection == "detected":
                    int_L_min = int_L_limit
                else:
                    int_L_max = int_L_limit
            if selection in ["detected", "undetected"] and self.min_flux_is_mesh:
                # mean over the cells of the slice, with the selection_function
                integral = self.mean_selected_luminosity_moment(int_L_limit, selection, 2)
            else:
                integral = self.integrate_luminosity_function_times_Lsq(
                    int_L_min, int_L_max)
            rho_L_second_moment = integral * self.luminosity_unit**2 / u.Mpc**3
            # I_lambda, I_nu or T_b per luminosity density, as for the mean intensity
            factor = self.intensity_per_luminosity_density(redshift)
            int_sec_mom = rho_L_second_moment * factor**2
            P_shot.append((int_sec_mom / self.mean_intensity**2).to(self.Mpch**3).value)
        return np.array(P_shot) * self.Mpch**3

    def get_intensity_shot_noise(self):
        """
        Calculates the shot noise (from randomly sampling the number of galaxies and the 
        luminosity of each galaxy), which is subject to smoothing from the second moment
        of the luminosity function: the mean over the slices of
        intensity_shot_noise_per_slice times the mean of the squared mask.

        """
        assert (len(self.redshift_mesh_axis) == self.obs_mask.shape[0])
        mask_sq = np.mean(self.obs_mask**2 * self.intensity_mask**2, axis=(1, 2))
        P_shot = np.mean(self.intensity_shot_noise_per_slice() * mask_sq)
        logging.info("intensity P_shot: {}".format(P_shot))
        return P_shot

    def cross_shot_noise_per_slice(self):
        """
        Poisson noise of the cross power spectrum in each slice along the LOS,
        without the mask, in (Mpc/h)^3: mean intensity of the galaxies that are
        in both maps / (mean intensity of the box * mean galaxy density of the
        galaxy map). The galaxy map is divided by its own mean density
        (galaxy_selection["n_gal"]), which includes galaxies that are not in
        the intensity map, e.g. for 'all' galaxies and 'detected' intensity.

        """
        galaxy_selections = self.galaxy_selection.values()
        if ("detected" in galaxy_selections) and ("undetected" in galaxy_selections):
            # no galaxy is in both maps
            return np.zeros(len(self.redshift_mesh_axis)) * self.Mpch**3
        elif "detected" in galaxy_selections:  # therefore 'undetected' is not present.
            galaxy_selection = "detected"
        elif "undetected" in galaxy_selections:  # therefore 'detected' is not present
            galaxy_selection = "undetected"
        else:  # both are 'all'
            galaxy_selection = "all"
        return (
            self.mean_intensity_per_redshift(
                self.redshift_mesh_axis, galaxy_selection=galaxy_selection
            )
            / self.mean_intensity
            / self.mean_intensity_per_redshift(
                self.redshift_mesh_axis,
                tracer="n_gal",
                galaxy_selection=self.galaxy_selection["n_gal"],
            )
        ).to(self.Mpch**3)

    def get_cross_shot_noise(self):
        """
        Calculates the shot noise of the cross power spectrum 
        that originates from randomly sampling the number of galaxies
        and their luminosity from the first moment of the luminosity function
        that includes only the galaxies overlapping in both fields:
        the mean over the slices of cross_shot_noise_per_slice times the mean
        of the product of the masks.

        """
        mask = np.mean(self.obs_mask**2 * self.intensity_mask * self.galaxy_mask, axis=(1, 2))
        return np.mean(self.cross_shot_noise_per_slice() * mask)

    def mean_intensity_per_redshift(self, redshifts, galaxy_selection="all", tracer="intensity"):
        """ As LognormalIntensityMock.mean_intensity_per_redshift, but the expected
        mean also if min_flux is a mesh: the model has no painted meshes to measure."""
        return super().mean_intensity_per_redshift(
            redshifts, galaxy_selection=galaxy_selection, tracer=tracer, use_mesh_means=False)

    @functools.cached_property
    def mean_intensity_per_redshift_mesh(self):
        """ Broadcasts the 1D mean_intensity_per_redshift array to the 3D mesh (cached)."""

        mean_intensity_per_redshift = self.mean_intensity_per_redshift(
            self.redshift_mesh_axis,
            galaxy_selection=self.galaxy_selection["intensity"],
        )
        mean_intensity_per_redshift_mesh = (
            np.ones(self.N_mesh) * mean_intensity_per_redshift[:, None, None]
        )
        return mean_intensity_per_redshift_mesh.to(self.mean_intensity)

    @functools.cached_property
    def mean_ngal_per_redshift_mesh(self):
        """ Broadcasts the 1D mean galaxy number density array to the 3D mesh (cached)."""

        mean_ngal_per_redshift = self.mean_intensity_per_redshift(
            self.redshift_mesh_axis,
            tracer="n_gal",
            galaxy_selection=self.galaxy_selection["n_gal"],
        )
        mean_ngal_per_redshift_mesh = (
            np.ones(self.N_mesh) * mean_ngal_per_redshift[:, None, None]
        )
        return mean_ngal_per_redshift_mesh

    @functools.cached_property
    def weight_mesh_im(self):
        """ Weights for the intensity mesh: just 1/mean_intensity."""

        return (1 / self.mean_intensity).to(1 / self.mean_intensity)

    @functools.cached_property
    def weight_mesh_ngal(self):
        """ Weights for the galaxy density mesh: 1/(mean n_gal as a function of position)."""
        return 1.0 / self.mean_ngal_per_redshift_mesh  # / self.n_bar_gal

    def get_intensity_model(self, sky_subtraction=False, return_3d=False):
        """
        Calculate and save the power spectrum model for the intensity.

        Parameters
        ----------
        sky_subtraction : bool
            Set to True if you want the sky_subtracted_intensity.
            Default: False.

        """
        logging.info("Getting intensity model.")
        sigma_noise_im = self.sigma_noise
        if callable(sigma_noise_im):
            sigma_noise_im = np.std(self.noise_mesh)
        damping_function = self.D_sq_par_of_k(
            self.k_par, self.s_par
        ) * self.D_sq_perp_of_k(
            self.k_perp, self.s_perp
        )  # * self.D_cic_corr**2
        if sky_subtraction:
            # (1 - D_sky)^2, Eq. 24 of the SIMPLE paper
            damping_function = damping_function * self.sky_subtraction_factor()**2
        mask_window_function = (
            self.obs_mask * self.intensity_mask
            * self.mean_intensity_per_redshift_mesh.to(self.mean_intensity)
            * self.weight_mesh_im
        ).to(1)
        noise = None
        if self.do_model_shot_noise:
            if self.im_shot_noise_model is None:
                self.model_shot_noise()
            P_shot_smoothed = self.im_shot_noise_model
        else:
            P_shot_smoothed = self.get_intensity_shot_noise()
            # the noise through the window, after the k_perp = 0 modes are removed
            noise = self.get_intensity_noise(sigma_noise_im, sky_subtraction)
        S_bar_im = self.get_S_bar(
            self.weight_mesh_im * self.intensity_mask * self.obs_mask, sigma_noise_im)
        if self.do_model_shot_noise and sky_subtraction:
            # the sky estimate contains the pixel noise, so S_bar V gets the factor too
            # (added after the window, as get_3d_pk_model does without noise)
            noise = (S_bar_im * self.box_volume * self.sky_subtraction_factor()**2
                     + P_shot_smoothed * damping_function) * ~self.removed_modes("intensity")
        logging.info("\nS_bar intensity: {}\n".format(
            (S_bar_im * self.box_volume).to(self.Mpch**3)))
        observed_volume = self.get_observed_volume()
        logging.info("Calculating multipoles.")
        if sky_subtraction:
            tracer = "sky_subtracted_intensity"
        else:
            tracer = "intensity"
        result_tuple = self.get_3d_pk_model(
            damping_function,
            P_shot_smoothed,
            S_bar_im,
            mask_window_function,
            observed_volume,
            self.box_volume,
            self.Mpch**3,
            convolve=True,
            save=True,
            tracer=tracer,
            return_3d=return_3d,
            noise=noise,
        )
        if return_3d:
            (model,
             mean_k,
             monopole,
             quadrupole,
             ) = result_tuple
        else:
            (
                mean_k,
                monopole,
                quadrupole,
            ) = result_tuple
        logging.info("Done.")
        return result_tuple

    def get_n_gal_model(self):
        """
        Calculate and save the power spectrum model for the galaxy number density.

        """

        logging.info("Getting n_gal model.")
        sigma_noise_ngal = np.sqrt(
            self.mean_ngal_per_redshift_mesh / self.voxel_volume
        )
        mean_ngal_per_redshift_mesh = self.mean_ngal_per_redshift_mesh  # lim.n_bar_gal
        mask_window_function = self.obs_mask * self.galaxy_mask
        # self.obs_mask * mean_ngal_per_redshift_mesh * self.weight_mesh_ngal
        damping_function = 1.0
        if self.do_model_shot_noise:
            if self.ngal_shot_noise_model is None:
                self.model_shot_noise()
            P_shot_smoothed = self.ngal_shot_noise_model
        else:
            # S_bar_ngal = self.get_S_bar(
            #    self.weight_mesh_ngal * self.obs_mask, sigma_noise_ngal)
            # P_shot_smoothed = S_bar_ngal * self.box_volume
            P_shot_smoothed = np.mean(
                (self.obs_mask * self.galaxy_mask)**2 / self.mean_ngal_per_redshift_mesh).to(self.Mpch**3)
            logging.info(
                "\n P_shot_ngal: {} \n".format(P_shot_smoothed))
        S_bar_notsmoothed = 0.0
        observed_volume = self.get_observed_volume()
        logging.info("Calculating multipoles.")
        (
            mean_k,
            monopole,
            quadrupole,
        ) = self.get_3d_pk_model(
            damping_function,
            P_shot_smoothed,
            S_bar_notsmoothed,
            mask_window_function,
            observed_volume,
            self.box_volume,
            self.Mpch**3,
            convolve=True,
            save=True,
            tracer="n_gal",
        )
        return mean_k, monopole, quadrupole

    def get_cross_model(self, sky_subtraction=False, return_3d = False):
        """
        Calculate and save the power spectrum model for the intensity-galaxy number density cross-power spectrum.

        Parameters
        ----------
        sky_subtraction : bool
            Set to True if you want the sky_subtracted_intensity.
            Default: False.

        """

        logging.info("Getting cross model.")
        mean_per_redshift_mesh_im = self.mean_intensity_per_redshift_mesh
        logging.info("Got mean intensity per redshift mesh.")
        mean_per_redshift_mesh_ngal = self.mean_ngal_per_redshift_mesh
        logging.info("Got mean ngal per redshift mesh.")

        mask_window_function_1 = (
            self.obs_mask * self.galaxy_mask * mean_per_redshift_mesh_ngal * self.weight_mesh_ngal
        )
        mask_window_function_2 = (
            self.obs_mask * self.intensity_mask * mean_per_redshift_mesh_im * self.weight_mesh_im
        )

        if self.do_model_shot_noise:
            if self.cross_shot_noise_model is None:
                self.model_shot_noise()
            P_shot_cross = self.cross_shot_noise_model
        else:
            P_shot_cross = self.get_cross_shot_noise()
        logging.info("Cross shot noise: {}".format(
            P_shot_cross.to(self.Mpch**3)))
        # self.n_bar_gal
        damping_function = np.sqrt(
            self.D_sq_par_of_k(self.k_par, self.s_par)
            * self.D_sq_perp_of_k(self.k_perp, self.s_perp)
        )  # self.D_cic_corr**2 *
        if sky_subtraction:
            # 1 - D_sky, Eq. 25 of the SIMPLE paper
            damping_function = damping_function * self.sky_subtraction_factor()
        noise = None
        if not self.do_model_shot_noise:
            # the noise through the windows, after the k_perp = 0 modes are removed
            noise = self.get_cross_noise(sky_subtraction)
        observed_volume = self.get_observed_volume()
        if sky_subtraction:
            tracer = "sky_subtracted_cross"
        else:
            tracer = "cross"
        logging.info("Calculating multipoles.")

        result_tuple= self.get_3d_pk_model(
            damping_function,
            P_shot_cross,
            0.0,
            mask_window_function_1,
            observed_volume,
            self.box_volume,
            self.Mpch**3,
            convolve=True,
            mask_window_function_2=mask_window_function_2,
            save=True,
            tracer=tracer,
            return_3d=return_3d,
            noise=noise,
        )
        if return_3d:
            (model,
             mean_k,
             monopole,
             quadrupole,
             ) = result_tuple
        else:
            (
                mean_k,
                monopole,
                quadrupole,
            ) = result_tuple
        return result_tuple

    def get_model(self, tracer):
        """ 
        Wrapper to calculate and save the model for each tracer.

        Parameters
        ----------
        tracer : str
            Tracer name to get the power spectrum model for.
            Options: "intensity", "n_gal", "cross", "sky_subtracted_intensity", "sky_subtracted_cross".

        """
        if tracer == "intensity":
            return self.get_intensity_model()
        elif tracer == "n_gal":
            return self.get_n_gal_model()
        elif tracer == "cross":
            return self.get_cross_model()
        elif tracer == "sky_subtracted_intensity":
            return self.get_intensity_model(sky_subtraction=True)
        elif tracer == "sky_subtracted_cross":
            return self.get_cross_model(sky_subtraction=True)
        else:
            raise ValueError("Tracer must be intensity, n_gal, or cross.")

    def get_models(self):
        """
        Wrapper to calculate and save all specified power spectrum models 
        here self.run_pk[tracer] is True.

        """
        if self.run_pk["intensity"]:
            self.get_model(tracer="intensity")
        if self.run_pk["n_gal"]:
            self.get_model(tracer="n_gal")
        if self.run_pk["cross"]:
            self.get_model(tracer="cross")
        if "sky_subtracted_intensity" in self.run_pk.keys():
            if self.run_pk["sky_subtracted_intensity"]:
                self.get_model(tracer='sky_subtracted_intensity')
        if "sky_subtracted_cross" in self.run_pk.keys():
            if self.run_pk["sky_subtracted_cross"]:
                self.get_model(tracer='sky_subtracted_cross')
