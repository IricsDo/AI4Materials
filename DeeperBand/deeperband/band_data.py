from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from BoltzTraP2 import sphere
import numpy as np
from pymatgen.io.vasp import Vasprun
from skimage import transform
import torch
from pymatgen.electronic_structure.bandstructure import BandStructure
import sys

class DFTData:
    """DFTData object used for BoltzTraP2 interpolation.

    Note that the units used by BoltzTraP are different to those used by VASP.

    Args:
        kpoints: The k-points in fractional coordinates.
        energies: The band energies in Hartree, formatted as (nbands, nkpoints).
        lattice_matrix: The lattice matrix in Bohr^3.
        mommat: The band structure derivatives.
    """

    def __init__(
        self,
        kpoints: np.ndarray,
        energies: np.ndarray,
        lattice_matrix: np.ndarray,
        mommat = None,
    ):
        self.kpoints = kpoints
        self.ebands = energies
        self.lattice_matrix = lattice_matrix
        self.volume = np.abs(np.linalg.det(self.lattice_matrix))
        self.mommat = mommat

    def get_lattvec(self) -> np.ndarray:
        """Get the lattice matrix. This method is required by BoltzTraP2."""
        return self.lattice_matrix

def kpoints_from_bandstructure(
    bandstructure: BandStructure, cartesian: bool = False
) -> np.ndarray:
    """Extract the k-points from a band structure.

    Args:
        bandstructure: A band structure object.
        cartesian: Whether to return the k-points in cartesian coordinates.

    Returns:
        A (n, 3) float array of the k-points.
    """
    if cartesian:
        kpoints = np.array([k.cart_coords for k in bandstructure.kpoints])
    else:
        kpoints = np.array([k.frac_coords for k in bandstructure.kpoints])

    return kpoints

def bands_fft(equiv, coeffs, lattvec, nworkers=1):
    """Rebuild the full energy bands from the interpolation coefficients.

    Adapted from BoltzTraP2.

    Args:
        equiv: list of k-point equivalence classes in direct coordinates
        coeffs: interpolation coefficients
        lattvec: lattice vectors of the system
        nworkers: number of working processes to span

    Returns:
        A 3-tuple (eband, vband) containing the energy bands  and group velocities.
        The shapes of those arrays are (nbands, nkpoints), (nbands, 3, nkpoints)
        where nkpoints is the total number of k points on the grid.
    """
    import multiprocessing as mp

    dallvec = np.vstack(equiv)
    sallvec = mp.sharedctypes.RawArray("d", dallvec.shape[0] * 3)
    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)
    dims = 2 * np.max(np.abs(dallvec), axis=0) + 1
    np.matmul(dallvec, lattvec.T, out=allvec)
    eband = np.zeros((len(coeffs), np.prod(dims)))
    vband = np.zeros((len(coeffs), 3, np.prod(dims)))

    # Span as many worker processes as needed, put all the bands in the queue,
    # and let them work until all the required FFTs have been computed.
    workers = []
    iq = mp.Queue()
    oq = mp.Queue()
    for iband, bandcoeff in enumerate(coeffs):
        iq.put((iband, bandcoeff))

    # The "None"s at the end of the queue signal the workers that there are
    # no more jobs left and they must therefore exit.
    for _i in range(nworkers):
        iq.put(None)

    for _i in range(nworkers):
        workers.append(mp.Process(target=worker, args=(equiv, sallvec, dims, iq, oq)))

    for w in workers:
        w.start()

    # The results of the FFTs are processed as soon as they are ready.
    for _r in range(len(coeffs)):
        iband, eband[iband], vband[iband] = oq.get()

    for w in workers:
        w.join()

    return eband.real, vband.transpose(0, 2, 1)


def worker(equivalences, sallvec, dims, iqueue, oqueue):
    """Thin wrapper around FFTev and FFTc to be used as a worker function.

    Adapted from BoltzTraP2.

    Args:
        equivalences: list of k-point equivalence classes in direct coordinates
        sallvec: Cartesian coordinates of all k points as a 1D vector stored
                    in shared memory.
        dims: upper bound on the dimensions of the k-point grid
        iqueue: input multiprocessing.Queue used to read bad indices
            and coefficients.
        oqueue: output multiprocessing.Queue where all results of the
            interpolation are put. Each element of the queue is a 4-tuple
            of the form (index, eband, vvband, cband), containing the band
            index, the energies, and the group velocities.

    Returns:
        None. The results of the calculation are put in oqueue.
    """
    from BoltzTraP2.fite import FFTev

    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)

    while True:
        task = iqueue.get()
        if task is None:
            break

        index, bandcoeff = task
        eband, vband = FFTev(equivalences, bandcoeff, allvec, dims)
        oqueue.put((index, eband, vband))

def sort_boltztrap_to_spglib(kpoints: np.ndarray) -> np.ndarray:
    """Get an index array that sorts the k-points from BoltzTraP2 to the spglib order.

    Args:
        kpoints: A (n, 3) float array of the k-points in fractional coordinates.

    Returns:
        A (n, ) int array of the sort order.
    """
    sort_idx = np.lexsort(
        (
            kpoints[:, 2],
            kpoints[:, 2] < 0,
            kpoints[:, 1],
            kpoints[:, 1] < 0,
            kpoints[:, 0],
            kpoints[:, 0] < 0,
        )
    )
    boltztrap_kpoints = kpoints[sort_idx]

    return np.lexsort(
        (
            boltztrap_kpoints[:, 0],
            boltztrap_kpoints[:, 0] < 0,
            boltztrap_kpoints[:, 1],
            boltztrap_kpoints[:, 1] < 0,
            boltztrap_kpoints[:, 2],
            boltztrap_kpoints[:, 2] < 0,
        )
    )
    
class FourierInterpolator:
    """Class to perform Fourier interpolation of electronic band structures.

    Interpolation is performed using BoltzTraP2.

    Args:
        band_structure: The Bandstructure object to be interpolated.
        magmom: Magnetic moments of the atoms.
        mommat: Momentum matrix, as supported by BoltzTraP2.
    """

    def __init__(
        self,
        band_structure: BandStructure,
        magmom = None,
        mommat = None,
    ):
        from BoltzTraP2.units import Angstrom
        from pymatgen.io.ase import AseAtomsAdaptor

        self._band_structure = band_structure
        self._spins = self._band_structure.bands.keys()
        self._lattice_matrix = band_structure.structure.lattice.matrix.T * Angstrom

        self._kpoints = kpoints_from_bandstructure(band_structure)
        self._atoms = AseAtomsAdaptor.get_atoms(band_structure.structure)

        self._magmom = magmom
        self._mommat = mommat
        self._structure = band_structure.structure

    def interpolate_bands(
        self,
        interpolation_factor: float = 5,
        return_velocities: bool = False,
        nworkers: int = -1,
    ):
        """Get an interpolated pymatgen band structure.

        Note, the interpolation mesh is determined using by ``interpolate_factor``
        option in the ``FourierInterpolator`` constructor.

        The degree of parallelization is controlled by the ``nworkers`` option.

        Args:
            interpolation_factor: The factor by which the band structure will
                be interpolated.
            return_velocities: Whether to return the group velocities.
            nworkers: The number of processors used to perform the
                interpolation. If set to ``-1``, the number of workers will
                be set to the number of CPU cores.

        Returns:
            The interpolated electronic structure. If ``return_velocities`` is True,
            the group velocities will also be returned as a  dict of
            ``{Spin: velocities}`` where velocities is a numpy array with the
            shape (nbands, nkpoints, 3) and has units of m/s.
        """
        import multiprocessing

        from BoltzTraP2 import fite, sphere
        from BoltzTraP2.units import eV
        from pymatgen.io.ase import AseAtomsAdaptor
        from scipy.constants import physical_constants
        try:
            import spglib as spglib
        except ImportError:
            from spglib import spglib   # fallback for very old installs

        coefficients = {}

        equivalences = sphere.get_equivalences(
            atoms=self._atoms,
            nkpt=self._kpoints.shape[0] * interpolation_factor,
            magmom=self._magmom,
        )

        # get the interpolation mesh used by BoltzTraP2
        interpolation_mesh = 2 * np.max(np.abs(np.vstack(equivalences)), axis=0) + 1

        for spin in self._spins:
            energies = self._band_structure.bands[spin] * eV
            data = DFTData(
                self._kpoints, energies, self._lattice_matrix, mommat=self._mommat
            )
            coefficients[spin] = fite.fitde3D(data, equivalences)

        nworkers = multiprocessing.cpu_count() if nworkers == -1 else nworkers

        energies = {}
        velocities = {}
        for spin in self._spins:
            energies[spin], velocities[spin] = bands_fft(
                equivalences,
                coefficients[spin],
                self._lattice_matrix,
                nworkers=nworkers,
            )

            # boltztrap2 gives energies in Rydberg, convert to eV
            energies[spin] /= eV

            # velocities in Bohr radius * Rydberg / hbar, convert to m/s.
            velocities[spin] *= (
                physical_constants["Bohr radius"][0]
                / physical_constants["atomic unit of time"][0]
            )

        efermi = self._band_structure.efermi

        atoms = AseAtomsAdaptor().get_atoms(self._band_structure.structure)
        atoms = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.numbers)
        mapping, grid = spglib.get_ir_reciprocal_mesh(
            interpolation_mesh, atoms, symprec=0.1
        )
        kpoints = grid / interpolation_mesh

        # sort energies so they have the same order as the k-points generated by spglib
        sort_idx = sort_boltztrap_to_spglib(kpoints)
        energies = {s: ener[:, sort_idx] for s, ener in energies.items()}
        velocities = {s: vel[:, sort_idx] for s, vel in velocities.items()}

        rlat = self._band_structure.structure.lattice.reciprocal_lattice
        interp_band_structure = BandStructure(
            kpoints, energies, rlat, efermi, structure=self._structure
        )

        if return_velocities:
            return interp_band_structure, velocities

        return interp_band_structure
    
def get_bands_sc(vasprunname,lenth=18):
    v = Vasprun(vasprunname)
    bs_uniform=v.get_band_structure()
    _kpoints = kpoints_from_bandstructure(bs_uniform)
    _atoms = AseAtomsAdaptor.get_atoms(bs_uniform.structure)
    print(_atoms.symbols.formula._formula)
    print(bs_uniform.structure)            
    print(f'>Spin Polarized: {bs_uniform.is_spin_polarized}')
    print(f'>Fermi Level: {bs_uniform.efermi:.4f} eV')
    equivalences = sphere.get_equivalences(
                atoms=_atoms,
                nkpt=_kpoints.shape[0] * 5,
                magmom=None
            )
    mesh = 2 * np.max(np.abs(np.vstack(equivalences)), axis=0) + 1

    # interpolate the energies onto a dense k-point mesh
    interpolator = FourierInterpolator(bs_uniform)

    dense_bs = interpolator.interpolate_bands()

    _spins=dense_bs.bands.keys()
    if(len(_spins)==1):
        sc_bands=dense_bs.bands[Spin(1)]
    else:
        sc_bands=np.vstack((dense_bs.bands[Spin(1)],dense_bs.bands[Spin(-1)]))

    #取最中间12条能带
    dense_bs_abs=np.zeros(len(sc_bands))
    if len(sc_bands)>lenth:
        for i in range(len(sc_bands)):
            dense_bs_abs[i]=sum(sc_bands[i]-dense_bs.efermi)
        argsorts=np.argsort(np.abs(dense_bs_abs))
        dense_bs_tmp=sc_bands[argsorts[:lenth]]-dense_bs.efermi
        dense_bs_abs=np.zeros(len(dense_bs_tmp))
        for i in range(len(dense_bs_tmp)):
            dense_bs_abs[i]=sum(dense_bs_tmp[i])
        argsorts=np.argsort(dense_bs_abs)

        #统一形状
        plot_z=dense_bs_tmp.reshape((-1,mesh[2],mesh[1],mesh[0]))
        bands_sc=np.zeros((lenth,32,32,32))
        for i in range(lenth):
            bands_sc[i]=transform.resize(plot_z[argsorts[i]],(32,32,32))
    else:
        bands_sc=np.zeros((lenth,32,32,32))
        startlen=(lenth-len(sc_bands))//2
        for i in range(len(sc_bands)):
            bands_sc[i+startlen]=transform.resize(sc_bands[i].reshape((mesh[2],mesh[1],mesh[0])),(32,32,32))
        
    return torch.tensor(bands_sc, dtype=torch.float).unsqueeze(dim=0)