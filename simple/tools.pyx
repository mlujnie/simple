cimport cython
from libc.math cimport floor
import numpy as np

ITYPE = int


def _as_plain_array(array, name, dtype=float):
    """
    Return `array` as a plain numpy array of `dtype`.

    The routines in this module work with bare numbers and assume that every
    length is expressed in the same unit. An astropy Quantity slipping through
    either crashes deep inside a loop with a confusing message about scalar
    conversion, or is silently misread, since np.asarray() drops units without
    converting them. Rejecting Quantities here turns both cases into one clear
    error at the call site.
    """
    if getattr(array, "unit", None) is not None:
        raise TypeError(
            "{0} must not carry units, but it is a Quantity in '{1}'. "
            "Strip the units at the call site, making sure all lengths use "
            "the same one, e.g. {0}.to(box_size.unit).value.".format(
                name, array.unit)
        )
    return np.asarray(array, dtype=dtype)

ctypedef fused index_t:
    int
    long long

ctypedef fused real_t:
    float
    double

cdef struct VoxelIndex:
    long long ix
    long long iy
    long long iz
    int flag          # 0 inside the box, 1 too high, -1 too low, -2 not finite

def _check_no_units(array, name):
    """Raise if `array` carries astropy units, without converting anything."""
    if getattr(array, "unit", None) is not None:
        raise TypeError(
            "{0} must not carry units, but it is a Quantity in '{1}'. "
            "Strip the units at the call site, making sure all lengths use "
            "the same one, e.g. {0}.to_value(box_size_unit).".format(
                name, array.unit))


def _as_contiguous_positions(Positions):
    """
    C-contiguous positions, as float32 or float64, whichever they already are.

    Anything else (float16, int, a list) becomes float64. The point is to avoid
    silently doubling the memory of a float32 catalog, which is what the
    lognormal_galaxies reader produces.
    """
    Positions = np.asarray(Positions)
    if Positions.dtype == np.float32:
        return np.ascontiguousarray(Positions, dtype=np.float32)
    return np.ascontiguousarray(Positions, dtype=np.float64)

@cython.cdivision(True)
cdef inline VoxelIndex _voxel_index(double x, double y, double z,
                                    double vx, double vy, double vz,
                                    long long nx, long long ny, long long nz) nogil:
    """
    Voxel index of one galaxy, wrapped into the periodic box.

    The single place where a position becomes a voxel index, used both when
    storing indices and when painting straight from positions, so the two can
    never drift apart. Declared inline, so the C compiler folds it into the
    calling loop at no cost.

    Returns the wrapped index plus a flag, leaving the caller to decide whether
    to count, ignore or raise on galaxies outside the box.
    """
    cdef VoxelIndex v
    cdef double px = x / vx
    cdef double py = y / vy
    cdef double pz = z / vz
    cdef long long ix, iy, iz

    # Casting a NaN, an infinity or a huge value to an integer is undefined in
    # C. NaN fails every comparison, so this one condition covers all three.
    if not (-1e18 < px < 1e18 and -1e18 < py < 1e18 and -1e18 < pz < 1e18):
        v.ix = 0
        v.iy = 0
        v.iz = 0
        v.flag = -2
        return v

    ix = <long long>floor(px)
    iy = <long long>floor(py)
    iz = <long long>floor(pz)

    if ix > nx - 1 or iy > ny - 1 or iz > nz - 1:
        v.flag = 1
    elif ix < 0 or iy < 0 or iz < 0:
        v.flag = -1
    else:
        v.flag = 0

    # C's % keeps the sign of the dividend, unlike Python's, so the extra
    # "+ n) % n" is needed to get a non-negative index.
    v.ix = ((ix % nx) + nx) % nx
    v.iy = ((iy % ny) + ny) % ny
    v.iz = ((iz % nz) + nz) % nz
    return v


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _fill_galaxy_indices(real_t[:, ::1] Positions,
                          long long[::1] N_mesh,
                          double[::1] Box_Size,
                          int[:, ::1] indices):
    """
    Typed inner loop of get_galaxy_indices_cython. Fills `indices` and returns
    the number of galaxies that fell outside the box, (too_high, too_low).

    Everything here is a C type, so the loop runs at C speed and uses no extra
    memory beyond `indices` itself.
    """
    cdef Py_ssize_t N_gal = Positions.shape[0]
    cdef Py_ssize_t i
    cdef double vx = Box_Size[0] / N_mesh[0]
    cdef double vy = Box_Size[1] / N_mesh[1]
    cdef double vz = Box_Size[2] / N_mesh[2]
    cdef long long nx = N_mesh[0]
    cdef long long ny = N_mesh[1]
    cdef long long nz = N_mesh[2]
    cdef VoxelIndex v
    cdef unsigned long long too_high = 0
    cdef unsigned long long too_low = 0

    for i in range(N_gal):
        v = _voxel_index(Positions[i, 0], Positions[i, 1], Positions[i, 2],
                         vx, vy, vz, nx, ny, nz)
        if v.flag == -2:
            raise ValueError(
                "Position of galaxy {} is not finite or is absurdly far "
                "outside the box: {}, {}, {}.".format(
                    i, Positions[i, 0], Positions[i, 1], Positions[i, 2]))
        elif v.flag == 1:
            too_high += 1
        elif v.flag == -1:
            too_low += 1

        indices[i, 0] = <int>v.ix
        indices[i, 1] = <int>v.iy
        indices[i, 2] = <int>v.iz

    return too_high, too_low


def get_galaxy_indices_cython(Positions,
                            N_mesh,
                            Box_Size):

    """
    Voxel indices (NGP assignment) of galaxies on a mesh.

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
    indices: ndarray
        Array of shape (N_gal, 3) of int32 voxel indices, wrapped into the
        periodic box.

    """

    _check_no_units(Positions, "Positions")
    Positions = _as_contiguous_positions(Positions)
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

    # The typed loop below needs C-contiguous arrays of exactly these dtypes.
    # These calls only copy when the input is not already in that form.
    Box_Size = np.ascontiguousarray(Box_Size, dtype=np.float64)
    N_mesh = np.ascontiguousarray(N_mesh, dtype=np.int64)

    if np.any(N_mesh > np.iinfo(np.int32).max):
        raise ValueError(
            "N_mesh = {} does not fit in the int32 voxel indices this "
            "function returns.".format(tuple(N_mesh)))

    cdef Py_ssize_t N_gal = Positions.shape[0]
    print("N_gal: ", N_gal)

    # int32 halves the memory of the index array: 1.2 GB instead of 2.4 GB
    # at 1e8 galaxies. No mesh dimension comes close to 2**31.
    indices = np.empty((N_gal, 3), dtype=np.int32)
    if Positions.dtype == np.float32:
        too_high, too_low = _fill_galaxy_indices[cython.float](
            Positions, N_mesh, Box_Size, indices)
    else:
        too_high, too_low = _fill_galaxy_indices[cython.double](
            Positions, N_mesh, Box_Size, indices)
    #too_high, too_low = _fill_galaxy_indices(Positions, N_mesh, Box_Size, indices)

    print("{} too high, {} too low out of {} (wrapped around the periodic box).".format(
        too_high, too_low, N_gal))
    return indices


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _paint_positions(real_t[:, ::1] Positions,
                      double[::1] Weights,
                      long long[::1] N_mesh,
                      double[::1] Box_Size,
                      double[:, :, ::1] mesh,
                     unsigned char[::1] mask,
                      bint use_mask):
    """
    Typed inner loop of catalog_to_mesh_cython: adds each galaxy's weight to
    its voxel. Returns (too_high, too_low), the number of galaxies that were
    outside the box and got wrapped around.
    """
    cdef Py_ssize_t N_gal = Positions.shape[0]
    cdef Py_ssize_t i
    cdef double vx = Box_Size[0] / N_mesh[0]
    cdef double vy = Box_Size[1] / N_mesh[1]
    cdef double vz = Box_Size[2] / N_mesh[2]
    cdef long long nx = N_mesh[0]
    cdef long long ny = N_mesh[1]
    cdef long long nz = N_mesh[2]
    cdef VoxelIndex v
    cdef unsigned long long too_high = 0
    cdef unsigned long long too_low = 0
    cdef Py_ssize_t j = 0

    for i in range(N_gal):
        if use_mask and mask[i] == 0:
            continue

        v = _voxel_index(Positions[i, 0], Positions[i, 1], Positions[i, 2],
                         vx, vy, vz, nx, ny, nz)
        if v.flag == -2:
            raise ValueError(
                "Position of galaxy {} is not finite or is absurdly far "
                "outside the box: {}, {}, {}.".format(
                    i, Positions[i, 0], Positions[i, 1], Positions[i, 2]))
        elif v.flag == 1:
            too_high += 1
        elif v.flag == -1:
            too_low += 1

        # because the mask is already applied to Weights
        mesh[v.ix, v.iy, v.iz] += Weights[j]
        j += 1

    return too_high, too_low


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _paint_indices(index_t[:, ::1] Indices,
                    double[::1] Weights,
                    long long[::1] N_mesh,
                    double[:, :, ::1] mesh,
                     unsigned char[::1] mask,
                      bint use_mask):
    """
    Typed inner loop of catalog_to_mesh_cython_use_indices. Bounds checking is
    off for speed, so the indices are checked explicitly: an out-of-range index
    would otherwise corrupt memory instead of raising.
    """
    cdef Py_ssize_t N_gal = Indices.shape[0]
    cdef Py_ssize_t i
    cdef long long nx = N_mesh[0]
    cdef long long ny = N_mesh[1]
    cdef long long nz = N_mesh[2]
    cdef long long ix, iy, iz
    cdef Py_ssize_t j = 0

    for i in range(N_gal):
        if use_mask and mask[i] == 0:
            continue

        ix = Indices[i, 0]
        iy = Indices[i, 1]
        iz = Indices[i, 2]
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            raise IndexError(
                "Voxel index of galaxy {} is ({}, {}, {}), outside a mesh of "
                "shape ({}, {}, {}). Were the indices computed for a different "
                "mesh?".format(i, ix, iy, iz, nx, ny, nz))

        # Weights is indexed by j because the mask is already applied to Weights
        mesh[ix, iy, iz] += Weights[j]
        j += 1


def catalog_to_mesh_cython(Positions,
                            Weights,
                            N_mesh,
                            Box_Size,
                            mask=None):

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
    mask: array-like, optional
        Boolean array of shape (N_gal,) selecting the galaxies to paint.
        Weights then holds one entry per selected galaxy. Passing a mask
        avoids building Positions[mask], a full copy of the catalog.

    Returns:
    --------
    mesh: ndarray
        Array of shape (N_x, N_y, N_z) representing the mesh with assigned weights.

    """

    _check_no_units(Positions, "Positions")
    Positions = _as_contiguous_positions(Positions)
    Weights = _as_plain_array(Weights, "Weights")
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

    Weights = np.ascontiguousarray(Weights, dtype=np.float64)
    Box_Size = np.ascontiguousarray(Box_Size, dtype=np.float64)
    N_mesh = np.ascontiguousarray(N_mesh, dtype=np.int64)

    cdef Py_ssize_t N_gal = Positions.shape[0]
    print("N_gal: ", N_gal)
    mask_view, use_mask = _as_mask(mask, N_gal, Weights.shape[0])

    mesh = np.zeros(tuple(N_mesh), dtype=np.float64)
    if Positions.dtype == np.float32:
        too_high, too_low = _paint_positions[cython.float](
            Positions, Weights, N_mesh, Box_Size, mesh, mask_view, use_mask)
    else:
        too_high, too_low = _paint_positions[cython.double](
            Positions, Weights, N_mesh, Box_Size, mesh, mask_view, use_mask)


    print("{} too high, {} too low out of {} (wrapped around the periodic box).".format(
        too_high, too_low, N_gal))
    return mesh


def catalog_to_mesh_cython_use_indices(Indices,
                            Weights,
                            N_mesh,
                            Box_Size,
                            mask=None):

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
        Size of the box enclosing the mesh, array of shape (3,). Not used, kept
        for backwards compatibility.
    mask: array-like, optional
        Boolean array of shape (N_gal,) selecting the galaxies to paint.
        Weights then holds one entry per selected galaxy. Passing a mask
        avoids building Indices[mask], a full copy of the index array.

    Returns:
    --------
    mesh: ndarray
        Array of shape (N_x, N_y, N_z) representing the mesh with assigned weights.

    """

    _check_no_units(Indices, "Indices")
    Weights = _as_plain_array(Weights, "Weights")
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

    # int32 and int64 indices are both used directly, so neither dtype costs
    # a conversion copy. Anything else becomes int32.
    Indices = np.asarray(Indices)
    if Indices.dtype == np.int64:
        Indices = np.ascontiguousarray(Indices, dtype=np.int64)
    else:
        Indices = np.ascontiguousarray(Indices, dtype=np.int32)

    Weights = np.ascontiguousarray(Weights, dtype=np.float64)
    N_mesh = np.ascontiguousarray(N_mesh, dtype=np.int64)

    cdef Py_ssize_t N_gal = Indices.shape[0]
    print("N_gal: ", N_gal)
    mask_view, use_mask = _as_mask(mask, N_gal, Weights.shape[0])

    mesh = np.zeros(tuple(N_mesh), dtype=np.float64)
    if Indices.dtype == np.int64:
        _paint_indices[cython.longlong](
            Indices, Weights, N_mesh, mesh, mask_view, use_mask)
    else:
        _paint_indices[cython.int](
            Indices, Weights, N_mesh, mesh, mask_view, use_mask)
    return mesh

def get_fratio_by_position(Positions,
                            Fluxes,
                            flux_limit_mesh,
                            N_mesh,
                            Box_Size):
    _check_no_units(Positions, "Positions")
    Positions = np.asarray(Positions)
    Fluxes = _as_plain_array(Fluxes, "Fluxes")
    flux_limit_mesh = _as_plain_array(flux_limit_mesh, "flux_limit_mesh")
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

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
    _check_no_units(Indices, "Indices")
    Indices = np.asarray(Indices)
    Fluxes = _as_plain_array(Fluxes, "Fluxes")
    flux_limit_mesh = _as_plain_array(flux_limit_mesh, "flux_limit_mesh")
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

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
    _check_no_units(Positions, "Positions")
    Positions = np.asarray(Positions)
    Fluxes = _as_plain_array(Fluxes, "Fluxes")
    flux_limit_mesh = _as_plain_array(flux_limit_mesh, "flux_limit_mesh")
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

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
    _check_no_units(Indices, "Indices")
    Indices = np.asarray(Indices)
    Fluxes = _as_plain_array(Fluxes, "Fluxes")
    flux_limit_mesh = _as_plain_array(flux_limit_mesh, "flux_limit_mesh")
    Box_Size = _as_plain_array(Box_Size, "Box_Size")

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

def _as_mask(mask, N_gal, N_weights, name="mask"):
    """
    Validate `mask` and return it as (uint8 view, use_mask).

    The mask selects which of the N_gal rows are painted; Weights holds one
    entry per selected row, so their lengths must agree. Returning a uint8
    view costs nothing: numpy bools are already one byte each.
    """
    if mask is None:
        if N_weights != N_gal:
            raise ValueError(
                "Weights has {} entries but there are {} galaxies and no "
                "mask.".format(N_weights, N_gal))
        return np.empty(0, dtype=np.uint8), False

    mask = np.ascontiguousarray(mask)
    if mask.dtype != np.bool_:
        raise TypeError("{} must be a boolean array, not {}.".format(
            name, mask.dtype))
    if mask.shape[0] != N_gal:
        raise ValueError(
            "{} has {} entries but there are {} galaxies.".format(
                name, mask.shape[0], N_gal))

    n_selected = int(np.count_nonzero(mask))
    if N_weights != n_selected:
        raise ValueError(
            "Weights has {} entries but the mask selects {} galaxies.".format(
                N_weights, n_selected))
    return mask.view(np.uint8), True
