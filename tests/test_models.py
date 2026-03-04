"""Tests for the SOP class and the Hubbard model Hamiltonian."""

import pytest
import numpy as np
from scipy import linalg as LA

try:
    import qiskit_nature
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# ---------------------------------------------------------------------------
# SOP initialization
# ---------------------------------------------------------------------------

class TestSOPInit:
    """SOP object stores residues, poles, and derived attributes correctly."""

    def test_scalar_sop(self):
        from sop_lake.SOP import SOP
        Gamma = [np.array([[1.0 + 0j]])]
        sigma = [0.0]
        sop = SOP(Gamma, sigma, p_type="std")
        assert sop.num_poles == 1
        assert sop.dim == 1
        assert sop.p_type == "std"

    def test_matrix_sop(self):
        from sop_lake.SOP import SOP
        Gamma = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
        sigma = [-1.0, 1.0]
        sop = SOP(Gamma, sigma)
        assert sop.num_poles == 2
        assert sop.dim == 2

    def test_sqrt_ptype(self):
        from sop_lake.SOP import SOP
        Gamma = [np.eye(2, dtype=complex)]
        sop = SOP(Gamma, [0.0], p_type="sqrt")
        assert sop.p_type == "sqrt"


# ---------------------------------------------------------------------------
# SOP evaluation
# ---------------------------------------------------------------------------

class TestSOPEvaluate:
    """G(ω) = Σ_k Γ_k / (ω − σ_k)."""

    def test_single_pole_std(self):
        """G(w) = Gamma / (w - sigma) for a single-pole SOP."""
        from sop_lake.SOP import SOP
        Gamma = np.array([[2.0 + 0j]])
        sigma = 1.0
        sop = SOP([Gamma], [sigma], p_type="std")
        w = 3.0
        result = sop.evaluate([w])
        expected = Gamma / (w - sigma)
        assert np.allclose(result[0], expected)

    def test_two_pole_cancellation(self):
        """G(0) = 1/(0-(-1)) + 1/(0-1) = 1 - 1 = 0."""
        from sop_lake.SOP import SOP
        G = [np.array([[1.0 + 0j]]), np.array([[1.0 + 0j]])]
        s = [-1.0, 1.0]
        sop = SOP(G, s, p_type="std")
        result = sop.evaluate([0.0])
        assert np.allclose(result[0], 0.0)

    def test_sqrt_ptype_evaluation(self):
        """sqrt parametrization: G(w) = Gamma @ Gamma / (w - sigma)."""
        from sop_lake.SOP import SOP
        Gamma = np.array([[2.0 + 0j]])
        sigma = 0.0
        sop = SOP([Gamma], [sigma], p_type="sqrt")
        w = 1.0
        result = sop.evaluate([w])
        expected = Gamma @ Gamma / (w - sigma)
        assert np.allclose(result[0], expected)

    def test_complex_frequency(self):
        """Evaluation works on the shifted real axis (complex w)."""
        from sop_lake.SOP import SOP
        Gamma = np.array([[1.0 + 0j]])
        sop = SOP([Gamma], [0.0], p_type="std")
        w = complex(1.0, -0.5)
        result = sop.evaluate([w])
        expected = Gamma / (w - 0.0)
        assert np.allclose(result[0], expected)

    def test_output_shape_multiple_freqs(self):
        """evaluate(w_list) should return shape (n_freqs, dim, dim)."""
        from sop_lake.SOP import SOP
        Gamma = np.eye(2, dtype=complex)
        sop = SOP([Gamma], [0.0], p_type="std")
        w_list = [1.0, 2.0, 3.0]
        result = sop.evaluate(w_list)
        assert result.shape == (3, 2, 2)

    def test_unknown_ptype_raises(self):
        from sop_lake.SOP import SOP
        sop = SOP([np.array([[1.0 + 0j]])], [0.0])
        sop.p_type = "bad"
        with pytest.raises(ValueError):
            sop.evaluate([1.0])


# ---------------------------------------------------------------------------
# SOP manipulations
# ---------------------------------------------------------------------------

class TestSOPManipulations:
    """sort, is_odd, to_dict, make_residues_hermitian, make_poles_real."""

    def test_sort_by_real_pole(self):
        from sop_lake.SOP import SOP
        G = [np.array([[1.0 + 0j]]), np.array([[2.0 + 0j]]), np.array([[3.0 + 0j]])]
        s = [2.0, -1.0, 0.5]
        sop = SOP(G, s)
        sop.sort()
        assert sop.sigma_list[0] == -1.0
        assert sop.sigma_list[1] == 0.5
        assert sop.sigma_list[2] == 2.0
        # Corresponding residue must follow
        assert np.allclose(sop.Gamma_list[0], np.array([[2.0 + 0j]]))

    def test_is_odd_true(self):
        """antisymm_SOP produces an odd spectrum."""
        from sop_lake.SOP import SOP, antisymm_SOP
        half_G = [np.array([[1.0 + 0j]])]
        half_s = [1.0]
        G, s = antisymm_SOP(half_G, half_s)
        sop = SOP(G, s)
        assert sop.is_odd() == True

    def test_is_odd_false(self):
        from sop_lake.SOP import SOP
        G = [np.array([[1.0 + 0j]]), np.array([[2.0 + 0j]])]
        sop = SOP(G, [-1.0, 1.0])
        assert sop.is_odd() == False

    def test_to_dict_keys(self):
        from sop_lake.SOP import SOP
        sop = SOP([np.eye(2, dtype=complex)], [0.5], p_type="std")
        d = sop.to_dict()
        assert set(d.keys()) == {"Gamma_list", "sigma_list", "p_type"}
        assert d["p_type"] == "std"

    def test_make_residues_hermitian(self):
        """Non-Hermitian residue is replaced by its Hermitian projection."""
        from sop_lake.SOP import SOP
        Gamma = np.array([[1.0 + 0j, 0.5 + 1j], [0.0 + 0j, 1.0 + 0j]])
        sop = SOP([Gamma.copy()], [0.0])
        sop.make_residues_hermitian()
        result = sop.Gamma_list[0]
        assert np.allclose(result, result.conj().T)

    def test_make_residues_pos_semidef_requires_std(self):
        """make_residues_pos_semidef raises for sqrt p_type."""
        from sop_lake.SOP import SOP
        sop = SOP([np.eye(2, dtype=complex)], [0.0], p_type="sqrt")
        with pytest.raises(ValueError):
            sop.make_residues_pos_semidef()


# ---------------------------------------------------------------------------
# SOP parameter flattening and reconstruction
# ---------------------------------------------------------------------------

class TestSOPParameters:
    """SOP_to_params / params_to_SOP round-trip."""

    def test_round_trip_1x1(self):
        from sop_lake.SOP import SOP_to_params, params_to_SOP
        G = [np.array([[1.0 + 0.5j]])]
        s = [complex(0.5, 0.0)]
        params = SOP_to_params(G, s)
        G_out, s_out = params_to_SOP(params, M=1)
        assert np.allclose(G_out[0], G[0])
        assert np.isclose(s_out[0], s[0])

    def test_round_trip_2x2(self):
        from sop_lake.SOP import SOP_to_params, params_to_SOP
        G = [np.array([[1.0 + 0j, 0.2 + 0.1j], [0.3 - 0.1j, 0.8 + 0j]])]
        s = [complex(0.5, 0.0)]
        params = SOP_to_params(G, s)
        G_out, s_out = params_to_SOP(params, M=1)
        assert np.allclose(G_out[0], G[0])

    def test_from_params_classmethod(self):
        from sop_lake.SOP import SOP, SOP_to_params
        G = [np.array([[2.0 + 0j]])]
        s = [1.0]
        params = SOP_to_params(G, s)
        sop = SOP.from_params(params, num_poles=1)
        assert np.allclose(sop.Gamma_list[0], G[0])
        assert np.isclose(sop.sigma_list[0], s[0])

    def test_get_params_and_from_params(self):
        """get_params() → from_params() round-trip."""
        from sop_lake.SOP import SOP
        G = [np.array([[2.0 + 1j]])]
        s = [complex(-0.5, 0.0)]
        sop = SOP(G, s)
        params = sop.get_params()
        sop2 = SOP.from_params(params, num_poles=1)
        assert np.allclose(sop2.Gamma_list[0], sop.Gamma_list[0])
        assert np.isclose(sop2.sigma_list[0], sop.sigma_list[0])


# ---------------------------------------------------------------------------
# adapt_residues
# ---------------------------------------------------------------------------

class TestAdaptResidues:

    def test_std_to_sqrt(self):
        from sop_lake.SOP import adapt_residues
        Gamma = np.array([[4.0 + 0j, 0.0 + 0j], [0.0 + 0j, 9.0 + 0j]])
        result = adapt_residues([Gamma], "std", "sqrt")
        expected = LA.sqrtm(Gamma)
        assert np.allclose(result[0], expected)

    def test_sqrt_to_std(self):
        from sop_lake.SOP import adapt_residues
        Gamma = np.array([[2.0 + 0j, 0.0 + 0j], [0.0 + 0j, 3.0 + 0j]])
        result = adapt_residues([Gamma], "sqrt", "std")
        expected = Gamma @ Gamma
        assert np.allclose(result[0], expected)

    def test_same_type_noop(self):
        from sop_lake.SOP import adapt_residues
        Gamma = np.eye(2, dtype=complex)
        result = adapt_residues([Gamma], "std", "std")
        assert np.allclose(result[0], Gamma)

    def test_invalid_type_raises(self):
        from sop_lake.SOP import adapt_residues
        with pytest.raises(ValueError):
            adapt_residues([np.eye(2, dtype=complex)], "std", "invalid")


# ---------------------------------------------------------------------------
# antisymm_SOP
# ---------------------------------------------------------------------------

class TestAntisymmSOP:

    def test_pole_antisymmetry(self):
        from sop_lake.SOP import antisymm_SOP
        half_G = [np.array([[1.0 + 0j]])]
        half_s = [2.0]
        G, s = antisymm_SOP(half_G, half_s)
        assert np.isclose(s[0], 2.0)
        assert np.isclose(s[1], -2.0)

    def test_residue_symmetry(self):
        from sop_lake.SOP import antisymm_SOP
        half_G = [np.array([[1.0 + 0j]]), np.array([[2.0 + 0j]])]
        half_s = [1.0, 2.0]
        G, s = antisymm_SOP(half_G, half_s)
        assert len(G) == 4
        assert np.allclose(G[0], G[3])
        assert np.allclose(G[1], G[2])

    def test_full_sop_is_odd(self):
        from sop_lake.SOP import SOP, antisymm_SOP
        half_G = [np.array([[0.5 + 0j]]), np.array([[0.3 + 0j]])]
        half_s = [0.5, 1.5]
        G, s = antisymm_SOP(half_G, half_s)
        sop = SOP(G, s)
        assert sop.is_odd() == True


# ---------------------------------------------------------------------------
# Hubbard model (requires qiskit_nature)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestHubbardModel:

    def test_single_site(self):
        from sop_lake.hubbard import Hubbarb_Ham_1D
        H = Hubbarb_Ham_1D(t=1.0, U=4.0, N=1, bc=0)
        assert H is not None

    def test_two_site_chain(self):
        from sop_lake.hubbard import Hubbarb_Ham_1D
        H = Hubbarb_Ham_1D(t=1.0, U=4.0, N=2, bc=0)
        assert H is not None

    def test_hopping_hamiltonian(self):
        from sop_lake.hubbard import hopping_Ham_1D
        h = hopping_Ham_1D(t=1.0, N=2, bc=0)
        assert h is not None

    def test_onsite_hamiltonian(self):
        from sop_lake.hubbard import onsite_Ham_1D
        H_onsite = onsite_Ham_1D(U=4.0, N=2)
        assert H_onsite is not None

    def test_prepare_small_system(self):
        """For N < 7 full system Hamiltonians are returned."""
        from sop_lake.hubbard import prepare_Hubbard_Hamiltonians
        h, H, hA, HA = prepare_Hubbard_Hamiltonians(t=1.0, U=4.0, N=4, NA=2, bc=0)
        assert H is not None
        assert h is not None
        assert HA is not None

    def test_prepare_large_system(self):
        """For N >= 7 full system Hamiltonians are returned as None."""
        from sop_lake.hubbard import prepare_Hubbard_Hamiltonians
        h, H, hA, HA = prepare_Hubbard_Hamiltonians(t=1.0, U=4.0, N=10, NA=2, bc=1)
        assert H is None
        assert h is None
        assert HA is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
