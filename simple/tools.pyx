import numpy as np

ITYPE = int
def catalog_to_mesh_cython(Positions,
                            Weights,
                            N_mesh,
                            Box_Size):

    """
    NGP assignment of galaxies with weights (e.g. intensity) to a mesh.

    Parameters:
    -----------
    Positions: array-like
        Array of shape (N_gal, 3) containing the positions of the galaxies.
    Weights: array-like
        Array of shape (N_gal,) containing the weights (e.g., intensity) of the galaxies.
    N_mesh: tuple
        Tuple of three integers (N_x, N_y, N_z) specifying the dimensions of the mesh.
    Box_Size: array-like
        Size of the box enclosing the mesh, array of shape (3,).

    Returns:
    --------
    mesh: ndarray
        Array of shape (N_x, N_y, N_z) representing the mesh with assigned weights.

    """

    cdef double[:,:,:] mesh
    mesh = np.zeros(N_mesh, dtype=float)
    voxel_size = Box_Size / N_mesh
    cdef unsigned long long int N_gal 
    N_gal = np.shape(Positions)[0]
    for i in range(N_gal):
        ix = int(np.floor(Positions[i,0] / voxel_size[0]))
        iy = int(np.floor(Positions[i,1] / voxel_size[1]))
        iz = int(np.floor(Positions[i,2] / voxel_size[2]))
        if ((ix > N_mesh[0] - 1) or (iy > N_mesh[1] - 1) or (iz > N_mesh[2] - 1)):
            continue
        mesh[ix, iy, iz] += Weights[i]
    return mesh

def apply_selection_function_by_position(Positions,
                            Fluxes,
                            flux_limit_mesh,
                            N_mesh,
                            Box_Size):
    cdef long[:] detected
    voxel_size = Box_Size / N_mesh
    cdef unsigned long long int N_gal 
    N_gal = np.shape(Positions)[0]
    detected = np.zeros(N_gal, dtype=int)
    for i in range(N_gal):
        ix = int(np.floor(Positions[i,0] / voxel_size[0]))
        iy = int(np.floor(Positions[i,1] / voxel_size[1]))
        iz = int(np.floor(Positions[i,2] / voxel_size[2]))
        ix = min([ix, N_mesh[0]-1])
        iy = min([iy, N_mesh[1]-1])
        iz = min([iz, N_mesh[2]-1])
        detected[i] = int(Fluxes[i] > flux_limit_mesh[ix, iy, iz])
        if i % 100000 == 0:
            print("Selection function: finished {}/{}.".format(i+1, N_gal))
    return detected


def getindep_cython(nx, ny, nz):
    """ From https://github.com/cblakeastro/intensitypower/tree/master."""

    indep = np.full((nx, ny, nz // 2 + 1), False, dtype=bool)
    indep[:, :, 1: nz // 2] = True
    indep[1: nx // 2, :, 0] = True
    indep[1: nx // 2, :, nz // 2] = True
    indep[0, 1: ny // 2, 0] = True
    indep[0, 1: ny // 2, nz // 2] = True
    indep[nx // 2, 1: ny // 2, 0] = True
    indep[nx // 2, 1: ny // 2, nz // 2] = True
    indep[nx // 2, 0, 0] = True
    indep[0, ny // 2, 0] = True
    indep[nx // 2, ny // 2, 0] = True
    indep[0, 0, nz // 2] = True
    indep[nx // 2, 0, nz // 2] = True
    indep[0, ny // 2, nz // 2] = True
    indep[nx // 2, ny // 2, nz // 2] = True
    return indep

def get_kspec_cython(int nx, int ny, int nz, float lx, float ly, float lz, dohalf=True, doindep=True):
    """
    Getting the wavenumber vectors k, their parallel and perpendicular components, their norm, and mu. From https://github.com/cblakeastro/intensitypower/tree/master.
    """

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    if dohalf:
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=lz / nz)[: nz // 2 + 1]
        indep = np.full((nx, ny, nz // 2 + 1), True, dtype=bool)
        if doindep:
            indep = getindep_cython(nx, ny, nz)
    else:
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=lz / nz)
        indep = np.full((nx, ny, nz), True, dtype=bool)
    indep[0, 0, 0] = False
    kspec = np.sqrt(
        kx[:, np.newaxis, np.newaxis] ** 2
        + ky[np.newaxis, :, np.newaxis] ** 2
        + kz[np.newaxis, np.newaxis, :] ** 2
    )
    kspec[0, 0, 0] = 1.0
    muspec = np.absolute(kx[:, np.newaxis, np.newaxis]) / kspec
    kspec[0, 0, 0] = 0.0

    k_par = kspec * muspec
    k_perp = kspec * np.sqrt(1 - muspec**2)

    return kspec, muspec, indep, kx, ky, kz, k_par, k_perp

def downsample_mask(old_array, long nx, long ny, long nz):
    """
    Downsample a 3D mask array by averaging neighboring elements.

    Parameters:
    -----------
    old_array: ndarray
        Array of shape (nx, ny, nz) representing the original mask.
    nx: int
        Number of elements along the x-axis in the original mask.
    ny: int
        Number of elements along the y-axis in the original mask.
    nz: int
        Number of elements along the z-axis in the original mask.

    Returns:
    --------
    new_array: ndarray
        Array of shape (nx // 2, ny // 2, nz // 2) representing the downsampled mask.

    """

    cdef long[:] new_shape
    new_shape = np.array([nx // 2, ny//2, nz//2])
    cdef double[:,:,:] new_array
    new_array = np.zeros(new_shape, dtype=float)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                new_array[i,j,k] = (old_array[2*i,2*j,2*k] \
                                    + old_array[2*i,2*j,2*k+1] \
                                    + old_array[2*i,2*j+1,2*k] \
                                    + old_array[2*i,2*j+1,2*k+1] \
                                    + old_array[2*i+1,2*j,2*k] \
                                    + old_array[2*i+1,2*j,2*k+1] \
                                    + old_array[2*i+1,2*j+1,2*k] \
                                    + old_array[2*i+1,2*j+1,2*k+1]) / 8
                #sum(old_array[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2])
    return new_array