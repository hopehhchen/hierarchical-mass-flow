from collections import defaultdict
import numpy as np
from astrodendro import Dendrogram, periodic_neighbours

import astropy.units as u
import astropy.constants as c
from skimage.morphology import binary_erosion
import pandas as pd

from .functions import *

class HMFlow3D(object):

    """
    A class object for calculating hierarchical mass flow.  The surfaces are
    defined using dendrogram.

    Parameters
    ------
    density: the density array sampled on a constant grid.  3D Numpy array.

    vx, vy, vz: the velocity arrays along the 0th, 1st and 2nd axes.
                3D Numpy arrays.

    pixscale: the size of the cell.  Astropy quantity.

    unit_density: the unit of the density array.  Astropy unit.  Deafult is
                  1/cm^3.  It can be either mass density or number density units.

    unit_velocity: the unit of the velocity arrays.  Astropy unit. Default is
                   km/s.
    """

    def __init__(self, density, vx, vy, vz, pixscale, unit_density = u.cm**-3., unit_velocity = u.km/u.s):

        # load the main data
        self.density = density
        self.vx, self.vy, self.vz = vx, vy, vz

        # load the scales
        self.pixscale = pixscale
        self.unit_density = unit_density
        self.unit_velocity = unit_velocity

        self.density_is_numberdensity = unit_density.is_equivalent(u.cm**-3.)
        ## Else assume that the density is a mass density.

    def dendrogram(self, periodic = True, **kwargs):

        """
        Calculate the dendrogram

        Parameters
        ------
        perodic: whether the arrays have a periodic boundary condition.  The
                 default is True.

        min_value: minimum value in density to consider.  See astrodendro.

        min_npix: minimum number of voxels in a structure for it to be
                  considered.  See astrodendro.

        min_delta: minimum difference in density for a structure to be
                   considered, i.e. from the saddle point where it joins with
                   another structure to the peak.  See astrodendro.
        """

        # test for non-optional parameters for astrodendro
        for k in ['min_value', 'min_npix', 'min_delta']:

            if k not in kwargs.keys():

                raise AttributeError('See astrodendro documentation.')

        # calculate dendrogram; indicate whether the box is periodic
        self.periodic = periodic

        neighbours = periodic_neighbours([0, 1, 2]) if self.periodic else None
        dendro = Dendrogram.compute(self.density, neighbours = neighbours, **kwargs)

        # output
        self.dendro = dendro
        print('Number of structures:', len(dendro))
        print('Number of leaves:', len(dendro.leaves))


    def calculate(self, direc = 'output.csv'):

        """
        Calculate the flux and the mass flow.

        Parameters
        ------
        direc: the directory to save the output table to.  Default is 'output.csv'
               in the local directory.
        """

        # this pre-process the density and velocity cubes into flux
        inflowcomp_magsign, flux_magsign, massflow_magsign = prepare3D(self)

        # calculation: loop through all dendrogram features.
        df_out = defaultdict(list)
        for i in range(len(self.dendro)):

            # obtain the mask for the structures and a mask for the surface
            mask_i = self.dendro[i].get_mask()
            mask_i_surf = (mask_i ^ binary_erosion(mask_i))

            # the three quantities
            df_out['inflow_surf'].append(np.median(inflowcomp_magsign[mask_i_surf]))
            df_out['flux_surf'].append(np.median(flux_magsign[mask_i_surf]))
            df_out['massflow_surf'].append(np.sum(massflow_magsign[mask_i_surf]))

            # effective radius: (V/(4*pi/3))^(1/3)
            df_out['Reff'].append(((np.sum(mask_i)*self.pixscale**3./(4.*np.pi/3.))**(1./3.)).to(u.pc).value)

        # output in csv
        df_out = pd.DataFrame(df_out)
        df_out.to_csv(direc, index = False)


class HMFlow2D(object):

    """
    A wrapper around the gradient fitting functions.  Mask out non-detection
    with NaN.  Make sure the maps are in the same shape and projection.

    Parameters
    ------
    N: column density/indensity map.

    v: velocity map.

    ev: an uncertainty map in the velocity fit.  This is used to weight the
        pixels when fitting the velocity gradient.  When None, use a uniform
        weighting instead.  Default is None.

    """


    def __init__(self, N, v, ev = None):

        self.N = N
        self.v = v
        self.ev = ev if ev is not None else None

        if ev is not None:
            self.mask = (np.isfinite(N) & np.isfinite(v) & np.isfinite(ev))
        else:
            self.mask = (np.isfinite(N) & np.isfinite(v))

    def grad_aperture(self, min_rad, max_rad, nstep_rad):

        """
        Circular aperture method.  Sizes are in units of pixel lengths.

        min_rad: the minimum radius.

        max_rad: the maximum radius.

        nstep_rad: how many steps to take to increase the radius of the aperture
                   from min_rad to max_rad.
        """

        xmesh, ymesh = np.meshgrid(np.arange(self.N.shape[1]),
                                   np.arange(self.N.shape[0]))

        x_iter, y_iter = xmesh[self.mask], ymesh[self.mask]
        rad_iter = np.linspace(min_rad, max_rad, int(nstep_rad))

        dict_massflow = defaultdict(list)
        df_massflow = defaultdict(list)
        for rad in rad_iter:

            map_massflow = np.zeros(self.N.shape)*np.nan

            for (x, y) in zip(x_iter, y_iter):

                mask_i = (np.hypot(xmesh-x, ymesh-y) <= rad)
                Sigma_i, dV_i, dR_i = quick2D(self.N, self.v, mask_i, ev = self.ev)
                ## Note that dV_i is velocity gradient (dV/dR).

                map_massflow[y, x] = Sigma_i*dV_i*(2.*dR_i)**2.

            # output maps for examination
            dict_massflow['size'].append(rad*2.)
            dict_massflow['map_massflow'].append(map_massflow)

            # calculate numbers for plotting the massflow-size relation
            df_massflow['med_massflow'].append(np.nanmedian(map_massflow))
            df_massflow['spr_massflow'].append(np.nanpercentile(map_massflow, 84.)-np.nanpercentile(map_massflow, 16.))
            ## Use the difference between 84th and 16th percentiles as an
            ## estimate for the spread.  This is based on 1-sig in a Gaussian
            ## distribution.

        df_massflow = pd.DataFrame(df_massflow)
        df_massflow['size'] = dict_massflow['size']
        self.df_massflow_aperture = df_massflow
        self.maps_massflow_aperture = dict_massflow

    def grad_dendro(self, min_value, min_npix, min_delta):

        """
        Calculate the dendrogram

        Parameters
        ------
        min_value: minimum value in density to consider.  See astrodendro.

        min_npix: minimum number of voxels in a structure for it to be
                  considered.  See astrodendro.

        min_delta: minimum difference in density for a structure to be
                   considered, i.e. from the saddle point where it joins with
                   another structure to the peak.  See astrodendro.
        """

        dendro = Dendrogram.compute(self.N, min_value = min_value, min_npix = min_npix, min_delta = min_delta)
        self.dendro = dendro

        df_massflow = defaultdict(list)
        for i in range(len(dendro)):

            mask_i = dendro[i].get_mask()

            Sigma_i, dV_i, dR_i = quick2D(self.N, self.v, mask_i, ev = self.ev)

            df_massflow['size'].append(2.*dR_i)
            df_massflow['massflow'].append(Sigma_i*dV_i*(2.*dR_i)**2.)

        df_massflow = pd.DataFrame(df_massflow)
        self.df_massflow_dendro = df_massflow
