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

        min_value:

        min_npix:

        min_delta:
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
