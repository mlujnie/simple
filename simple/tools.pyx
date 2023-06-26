import numpy as np

ITYPE = int
def catalog_to_mesh_cython(Positions,
                            Weights,
                            N_mesh,
                            Box_Size):
    """ NGP assignment of galaxies with weights (e.g. intensity) to a mesh."""
    cdef double[:,:,:] mesh
    mesh = np.zeros(N_mesh, dtype=float)
    voxel_size = Box_Size / N_mesh
    N_gal = np.shape(Positions)[0]
    for i in range(N_gal):
        ix = int(np.floor(Positions[i,0] / voxel_size[0]))
        iy = int(np.floor(Positions[i,1] / voxel_size[1]))
        iz = int(np.floor(Positions[i,2] / voxel_size[2]))
        mesh[ix, iy, iz] += Weights[i]
        if i % 100000 == 0:
            print("Finished {}/{}.".format(i+1, N_gal))
    return mesh

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