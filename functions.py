import numpy as np

import astropy.units as u
import astropy.constants as c

# constants
mass_avg = 2.8*u.u

def prepare3D(HMFlow_object):

    """
    Function for array manipulations to prepare for the calculation of the flux.


    """

    # load the data in the HMFlow_object (a HMFlow3D object)
    density = HMFlow_object.density
    vx, vy, vz = HMFlow_object.vx, HMFlow_object.vy, HMFlow_object.vz
    pixscale = HMFlow_object.pixscale
    unit_density, unit_velocity = HMFlow_object.unit_density, HMFlow_object.unit_velocity

    # obtain the shape of the cubes and the density scaling.
    dimensions = density.shape ## assuming that the cubes have the same shape.
    density_scale = mass_avg if HMFlow_object.density_is_numberdensity else 1.

    # density gradient as a proxy for normal directions
    if HMFlow_object.periodic:

        ## this is for boxes with periodic boundary conditions ##

        density_rep = np.zeros((dimensions[0]+2, dimensions[1]+2, dimensions[1]+2))*np.nan
        # the inner box
        density_rep[1:dimensions[0]+1, 1:dimensions[1]+1, 1:dimensions[2]+1] = density
        # 6 faces
        density_rep[0, 1:dimensions[1]+1, 1:dimensions[2]+1] = density[-1, :, :]
        density_rep[1:dimensions[0]+1, 0, 1:513] = density[:, -1, :]
        density_rep[1:dimensions[0]+1, 1:dimensions[1]+1, 0] = density[:, :, -1]
        density_rep[-1, 1:dimensions[1]+1, 1:513] = density[0, :, :]
        density_rep[1:dimensions[0]+1, -1, 1:513] = density[:, 0, :]
        density_rep[1:dimensions[0]+1, 1:dimensions[1]+1, -1] = density[:, :, 0]
        # 12 edges
        density_rep[0, 0, 1:dimensions[2]+1] = density[-1, -1, :]
        density_rep[0, -1, 1:dimensions[2]+1] = density[-1, 0, :]
        density_rep[-1, 0, 1:dimensions[2]+1] = density[0, -1, :]
        density_rep[-1, -1, 1:dimensions[2]+1] = density[0, 0, :]
        density_rep[0, 1:dimensions[1]+1, 0] = density[-1, :, -1]
        density_rep[0, 1:dimensions[1]+1, -1] = density[-1, :, 0]
        density_rep[-1, 1:dimensions[1]+1, 0] = density[0, :, -1]
        density_rep[-1, 1:dimensions[1]+1, -1] = density[0, :, 0]
        density_rep[1:dimensions[0]+1, 0, 0] = density[:, -1, -1]
        density_rep[1:dimensions[0]+1, 0, -1] = density[:, -1, 0]
        density_rep[1:dimensions[0]+1, -1, 0] = density[:, 0, -1]
        density_rep[1:dimensions[0]+1, -1, -1] = density[:, 0, 0]
        # 8 corners
        density_rep[0, 0, 0] = density[-1, -1, -1]
        density_rep[-1, 0, 0] = density[0, -1, -1]
        density_rep[0, -1, 0] = density[-1, 0, -1]
        density_rep[0, 0, -1] = density[-1, -1, 0]
        density_rep[-1, -1, 0] = density[0, 0, -1]
        density_rep[-1, 0, -1] = density[0, -1, 0]
        density_rep[0, -1, -1] = density[-1, 0, 0]
        density_rep[-1, -1, -1] = density[0, 0, 0]

        # gradient
        density_grad_x, density_grad_y, density_grad_z = np.gradient(density_rep)
        density_grad_x = density_grad_x[1:-1, 1:-1, 1:-1]
        density_grad_y = density_grad_y[1:-1, 1:-1, 1:-1]
        density_grad_z = density_grad_z[1:-1, 1:-1, 1:-1]

    else:

        # gradient
        ## Watch out for potentially weird behaviors toward the edges of the box.
        density_grad_x, density_grad_y, density_grad_z = np.gradient(density)

    density_grad_mag = np.sqrt(density_grad_x**2.+density_grad_y**2.+density_grad_z**2.)

    # inflow velocity, flux and mass flow: magnitudes with signs
    _dot_product = density_grad_x*vx + density_grad_y*vy + density_grad_z*vz
    inflowcomp_magsign = _dot_product/density_grad_mag

    # the output; in units of solar masses, pc and years.
    flux_magsign = ((inflowcomp_magsign*unit_velocity)*(density*unit_density)*density_scale).to(u.Msun*u.pc**-2.*u.yr**-1.).value
    massflow_magsign = (flux_magsign*(u.Msun*u.pc**-2.*u.yr**-1.)*pixscale**2.).to(u.Msun*u.yr**-1.).value

    return inflowcomp_magsign, flux_magsign, massflow_magsign
