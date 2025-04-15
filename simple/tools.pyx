import numpy as np

ITYPE = int

def get_galaxy_indices_cython(Positions,
                            N_mesh,
                            Box_Size):

    """
    NGP assignment of galaxies with weights (e.g. intensity) to a mesh.

    Parameters:
    -----------
    Positions: array-like
        Array of shape (N_gal, 3) containing the positions of the galaxies.
    N_mesh: tuple
        Tuple of three integers (N_x, N_y, N_z) specifying the dimensions of the mesh.
    Box_Size: array-like
        Size of the box enclosing the mesh, array of shape (3,).

    Returns:
    --------
    mesh: ndarray
        Array of shape (N_x, N_y, N_z) representing the mesh with assigned weights.

    """

    voxel_size = Box_Size / N_mesh
    cdef unsigned long long int N_gal 
    N_gal = np.shape(Positions)[0]
    print("N_gal: ", N_gal)

    cdef long[:,:] indices
    indices = np.zeros((N_gal, 3), dtype=int) 

    too_highs = 0
    too_lows = 0
    for i in range(N_gal):
        ix = int( (Positions[i,0] // voxel_size[0]) % N_mesh[0] )
        iy = int( (Positions[i,1] // voxel_size[1]) % N_mesh[1] )
        iz = int( (Positions[i,2] // voxel_size[2]) % N_mesh[2] )
        if ((ix > N_mesh[0] - 1) or (iy > N_mesh[1] - 1) or (iz > N_mesh[2] - 1)):
            too_highs += 1
            continue
        if (ix < 0) or (iy < 0) or (iz < 0):
            too_lows += 1
            continue
        indices[i][0] = ix
        indices[i][1] = iy
        indices[i][2] = iz
    print("{} too high, {} too low out of {}.".format(too_highs, too_lows, N_gal))
    return indices

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
    print("N_gal: ", N_gal)
    assert np.shape(Weights)[0] == N_gal
    too_highs = 0
    too_lows = 0
    for i in range(N_gal):
        ix = int(np.floor(Positions[i,0] / voxel_size[0])) % N_mesh[0]
        iy = int(np.floor(Positions[i,1] / voxel_size[1])) % N_mesh[1]
        iz = int(np.floor(Positions[i,2] / voxel_size[2])) % N_mesh[2]
        mesh[ix, iy, iz] += Weights[i]
    print("{} too high, {} too low out of {}.".format(too_highs, too_lows, N_gal))
    return mesh

def catalog_to_mesh_cython_use_indices(Indices,
                            Weights,
                            N_mesh,
                            Box_Size):

    """
    NGP assignment of galaxies with weights (e.g. intensity) to a mesh.

    Parameters:
    -----------
    Indices: array-like
        Array of shape (N_gal, 3) containing the voxel indices of the galaxies.
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
    N_gal = np.shape(Indices)[0]
    print("N_gal: ", N_gal)
    assert np.shape(Weights)[0] == N_gal
    too_highs = 0
    too_lows = 0
    for i in range(N_gal):
        ix = Indices[i,0]
        iy = Indices[i,1]
        iz = Indices[i,2]
        mesh[ix, iy, iz] += Weights[i]
    print("{} too high, {} too low out of {}.".format(too_highs, too_lows, N_gal))
    return mesh

def get_fratio_by_position(Positions,
                            Fluxes,
                            flux_limit_mesh,
                            N_mesh,
                            Box_Size):
    cdef long[:] detected
    voxel_size = Box_Size / N_mesh
    cdef unsigned long long int N_gal 
    N_gal = np.shape(Positions)[0]
    fratios = np.zeros(N_gal, dtype=float)
    for i in range(N_gal):
        ix = int(np.floor(Positions[i,0] / voxel_size[0])) % N_mesh[0]
        iy = int(np.floor(Positions[i,1] / voxel_size[1])) % N_mesh[1]
        iz = int(np.floor(Positions[i,2] / voxel_size[2])) % N_mesh[2]
        fratios[i] = Fluxes[i] / flux_limit_mesh[ix, iy, iz]
        if i % 100000 == 0:
            print("Selection function: finished {}/{}.".format(i+1, N_gal))
    return fratios

def get_fratio_by_position_use_indices(Indices,
                            Fluxes,
                            flux_limit_mesh,
                            N_mesh,
                            Box_Size):
    cdef long[:] detected
    voxel_size = Box_Size / N_mesh
    cdef unsigned long long int N_gal 
    N_gal = np.shape(Indices)[0]
    fratios = np.zeros(N_gal, dtype=float)
    for i in range(N_gal):
        ix = Indices[i,0]
        iy = Indices[i,1]
        iz = Indices[i,2]
        fratios[i] = Fluxes[i] / flux_limit_mesh[ix, iy, iz]
        if i % 100000 == 0:
            print("Selection function: finished {}/{}.".format(i+1, N_gal))
    return fratios

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
        ix = int(np.floor(Positions[i,0] / voxel_size[0])) % N_mesh[0]
        iy = int(np.floor(Positions[i,1] / voxel_size[1])) % N_mesh[1]
        iz = int(np.floor(Positions[i,2] / voxel_size[2])) % N_mesh[2]
        detected[i] = int(Fluxes[i] > flux_limit_mesh[ix, iy, iz])
        if i % 100000 == 0:
            print("Selection function: finished {}/{}.".format(i+1, N_gal))
    return detected

def apply_selection_function_by_position_use_indices(Indices,
                            Fluxes,
                            flux_limit_mesh,
                            N_mesh,
                            Box_Size):
    cdef long[:] detected
    voxel_size = Box_Size / N_mesh
    cdef unsigned long long int N_gal 
    N_gal = np.shape(Indices)[0]
    detected = np.zeros(N_gal, dtype=int)
    for i in range(N_gal):
        ix = Indices[i,0]
        iy = Indices[i,1]
        iz = Indices[i,2]
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
