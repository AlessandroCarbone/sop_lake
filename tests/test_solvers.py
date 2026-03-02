"""Tests for utility functions, embedding utilities, and the Lanczos algorithm."""

import sys
import os
import pytest
import numpy as np
from scipy import linalg as LA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import qiskit_nature
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# ---------------------------------------------------------------------------
# utils.py — matrix utilities
# ---------------------------------------------------------------------------

class TestCheckSelfAdjoint:

    def test_hermitian_matrix(self):
        from utils import check_selfadjoint
        A = np.array([[2.0 + 0j, 1.0 - 1j], [1.0 + 1j, 3.0 + 0j]])
        assert check_selfadjoint(A) == True

    def test_non_hermitian_matrix(self):
        from utils import check_selfadjoint
        A = np.array([[1.0 + 0j, 1.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        assert check_selfadjoint(A) == False

    def test_real_symmetric_matrix(self):
        from utils import check_selfadjoint
        A = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=complex)
        assert check_selfadjoint(A) == True

    def test_identity_is_hermitian(self):
        from utils import check_selfadjoint
        assert check_selfadjoint(np.eye(4, dtype=complex)) == True


class TestClosestHermitian:

    def test_projects_to_hermitian(self):
        from utils import closest_hermitian
        A = np.array([[1.0 + 0j, 0.5 + 1j], [0.0 + 0j, 2.0 + 0j]])
        H = closest_hermitian(A)
        assert np.allclose(H, H.conj().T)

    def test_is_average_of_A_and_Adag(self):
        from utils import closest_hermitian
        A = np.array([[1.0 + 0j, 0.5 + 1j], [0.0 + 0j, 2.0 + 0j]])
        H = closest_hermitian(A)
        expected = 0.5 * (A + A.conj().T)
        assert np.allclose(H, expected)

    def test_hermitian_unchanged(self):
        from utils import closest_hermitian
        A = np.array([[1.0 + 0j, 1.0 - 1j], [1.0 + 1j, 2.0 + 0j]])
        H = closest_hermitian(A)
        assert np.allclose(H, A)


class TestIsPosSemidef:

    def test_positive_definite(self):
        from utils import is_pos_semidef
        A = np.array([[2.0 + 0j, 0.0], [0.0, 3.0 + 0j]])
        assert is_pos_semidef(A) == True

    def test_zero_eigenvalue_psd(self):
        """A PSD matrix with a zero eigenvalue should return True."""
        from utils import is_pos_semidef
        A = np.array([[1.0 + 0j, 0.0], [0.0, 0.0 + 0j]])
        assert is_pos_semidef(A) == True

    def test_negative_eigenvalue(self):
        from utils import is_pos_semidef
        A = np.array([[1.0 + 0j, 0.0], [0.0, -1.0 + 0j]])
        assert is_pos_semidef(A) == False

    def test_non_hermitian_flagged(self):
        from utils import is_pos_semidef
        A = np.array([[1.0 + 0j, 1.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        result = is_pos_semidef(A)
        # Returns the string "Not Hermitian" for non-Hermitian inputs
        assert result == "Not Hermitian"


class TestClosestPosSemidef:

    def test_psd_after_projection(self):
        from utils import closest_pos_semidef
        A = np.array([[1.0 + 0j, 0.0], [0.0, -0.5 + 0j]])
        B = closest_pos_semidef(A)
        eigvals = LA.eigh(B)[0]
        assert np.all(eigvals >= -1e-10)

    def test_psd_matrix_unchanged(self):
        """A PSD matrix should be unchanged by projection."""
        from utils import closest_pos_semidef
        A = np.array([[2.0 + 0j, 0.0], [0.0, 1.0 + 0j]])
        B = closest_pos_semidef(A)
        assert np.allclose(B, A)

    def test_non_hermitian_raises(self):
        from utils import closest_pos_semidef
        A = np.array([[1.0 + 0j, 1.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        with pytest.raises(ValueError):
            closest_pos_semidef(A)


# ---------------------------------------------------------------------------
# utils.py — numerical utilities
# ---------------------------------------------------------------------------

class TestRMSE:

    def test_identical_lists(self):
        from utils import RMSE
        x = [1.0, 2.0, 3.0, 4.0]
        assert np.isclose(RMSE(x, x), 0.0)

    def test_known_value(self):
        """RMSE([0,0], [3,4]) = sqrt((9+16)/2) = sqrt(12.5)."""
        from utils import RMSE
        assert np.isclose(RMSE([0.0, 0.0], [3.0, 4.0]), np.sqrt(12.5))

    def test_single_element(self):
        from utils import RMSE
        assert np.isclose(RMSE([5.0], [2.0]), 3.0)


class TestMatsubaraFreq:

    def test_n_zero(self):
        """ω_0 = π/β."""
        from utils import Matsubara_freq
        beta = 100.0
        assert np.isclose(Matsubara_freq(0, beta), np.pi / beta)

    def test_n_one(self):
        """ω_1 = 3π/β."""
        from utils import Matsubara_freq
        beta = 50.0
        assert np.isclose(Matsubara_freq(1, beta), 3 * np.pi / beta)

    def test_fermionic_formula(self):
        """ω_n = (2n+1)π/β for all n."""
        from utils import Matsubara_freq
        beta = 200.0
        for n in [0, 1, 5, 10, 100]:
            expected = (2 * n + 1) * np.pi / beta
            assert np.isclose(Matsubara_freq(n, beta), expected)

    def test_negative_n(self):
        """ω_{-1} = -π/β."""
        from utils import Matsubara_freq
        beta = 100.0
        assert np.isclose(Matsubara_freq(-1, beta), -np.pi / beta)


class TestNumericalGrad:

    def test_linear_function(self):
        """∇(a·x) = a."""
        from utils import numerical_grad
        a = np.array([2.0, -3.0, 1.0])
        f = lambda x: float(np.dot(a, x))
        grad = numerical_grad(f, np.zeros(3), delta=1e-5)
        assert np.allclose(grad, a, atol=1e-7)

    def test_quadratic_function(self):
        """∇(x·x) = 2x."""
        from utils import numerical_grad
        f = lambda x: float(np.dot(x, x))
        x0 = np.array([1.0, 2.0, 3.0])
        grad = numerical_grad(f, x0, delta=1e-5)
        assert np.allclose(grad, 2 * x0, atol=1e-5)

    def test_scalar_function(self):
        """∇(x[0]^2) w.r.t. x[0] = 2*x[0]."""
        from utils import numerical_grad
        f = lambda x: x[0] ** 2
        x0 = np.array([3.0])
        grad = numerical_grad(f, x0, delta=1e-5)
        assert np.isclose(grad[0], 6.0, atol=1e-5)


# ---------------------------------------------------------------------------
# embedding_utils.py — frequency axis construction
# ---------------------------------------------------------------------------

class TestFrequencyAxis:

    def test_shift_axis_length(self):
        from embedding_utils import frequency_axis
        w_list, w_sim = frequency_axis("shift", eta_axis=0.5, num_pts=200, w_edges=[-5, 5])
        assert len(w_list) == 200
        assert len(w_sim) == 200

    def test_shift_axis_imaginary_part(self):
        """Shifted axis: Im(w_sim) = -η for all points."""
        from embedding_utils import frequency_axis
        eta = 0.5
        w_list, w_sim = frequency_axis("shift", eta_axis=eta, num_pts=100, w_edges=[-5, 5])
        for ws in w_sim:
            assert np.isclose(ws.imag, -eta)

    def test_erf_axis_length(self):
        from embedding_utils import frequency_axis
        w_list, w_sim = frequency_axis("erf", eta_axis=0.5, num_pts=150, w_edges=[-5, 5])
        assert len(w_sim) == 150

    def test_erf_axis_nonzero_imag_away_from_zero(self):
        """ERF axis: points away from zero should have non-zero imaginary part."""
        from embedding_utils import frequency_axis
        w_list, w_sim = frequency_axis("erf", eta_axis=1.0, num_pts=100, w_edges=[-5, 5])
        # The point furthest from zero should have a large imaginary part
        max_shift = max(abs(ws.imag) for ws in w_sim)
        assert max_shift > 0.5

    def test_matsubara_axis_purely_imaginary(self):
        """Matsubara frequencies should lie on the imaginary axis."""
        from embedding_utils import frequency_axis
        _, w_sim = frequency_axis(
            "imaginary", eta_axis=0.0,
            matsubara_params={"beta": 10.0, "Nw_max": 5}
        )
        for ws in w_sim:
            assert np.isclose(ws.real, 0.0)

    def test_matsubara_count(self):
        """Should return 2*Nw_max Matsubara frequencies."""
        from embedding_utils import frequency_axis
        Nw = 10
        w_list, w_sim = frequency_axis(
            "imaginary", eta_axis=0.0,
            matsubara_params={"beta": 100.0, "Nw_max": Nw}
        )
        assert len(w_list) == 2 * Nw

    def test_matsubara_spacing(self):
        """Consecutive Matsubara frequencies differ by 2π/β."""
        from embedding_utils import frequency_axis
        beta = 50.0
        w_list, _ = frequency_axis(
            "imaginary", eta_axis=0.0,
            matsubara_params={"beta": beta, "Nw_max": 5}
        )
        spacing = 2 * np.pi / beta
        for i in range(len(w_list) - 1):
            assert np.isclose(w_list[i + 1] - w_list[i], spacing)


# ---------------------------------------------------------------------------
# embedding_utils.py — linear mixing
# ---------------------------------------------------------------------------

class TestLinearMixing:

    def test_alpha_one_returns_list1(self):
        from embedding_utils import linear_mixing_lists
        list1 = [1.0, 2.0, 3.0]
        list2 = [4.0, 5.0, 6.0]
        result = linear_mixing_lists(list1, list2, alpha=1.0)
        assert np.allclose(result, list1)

    def test_equal_mixing(self):
        from embedding_utils import linear_mixing_lists
        list1 = [2.0, 4.0]
        list2 = [0.0, 0.0]
        result = linear_mixing_lists(list1, list2, alpha=0.5)
        assert np.allclose(result, [1.0, 2.0])

    def test_mixing_formula(self):
        """result = α*list1 + (1-α)*list2."""
        from embedding_utils import linear_mixing_lists
        alpha = 0.3
        list1 = [10.0, 20.0]
        list2 = [0.0, 0.0]
        result = linear_mixing_lists(list1, list2, alpha=alpha)
        assert np.allclose(result, [alpha * 10.0, alpha * 20.0])

    def test_mixing_numpy_arrays(self):
        """Works element-wise with numpy arrays."""
        from embedding_utils import linear_mixing_lists
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.zeros((2, 2))
        result = linear_mixing_lists([A], [B], alpha=0.5)
        assert np.allclose(result[0], 0.5 * A)

    def test_unequal_length_raises(self):
        from embedding_utils import linear_mixing_lists
        with pytest.raises(ValueError):
            linear_mixing_lists([1.0, 2.0], [1.0], alpha=0.5)

    def test_alpha_zero_raises(self):
        from embedding_utils import linear_mixing_lists
        with pytest.raises(ValueError):
            linear_mixing_lists([1.0], [2.0], alpha=0.0)

    def test_alpha_too_large_raises(self):
        from embedding_utils import linear_mixing_lists
        with pytest.raises(ValueError):
            linear_mixing_lists([1.0], [2.0], alpha=1.5)


# ---------------------------------------------------------------------------
# embedding_utils.py — DOS difference
# ---------------------------------------------------------------------------

class TestDOSDiff:

    def test_identical_greens_functions(self):
        from embedding_utils import DOS_diff
        ntot = 2
        G = [np.eye(ntot, dtype=complex) * complex(0, -0.5) for _ in range(50)]
        assert np.isclose(DOS_diff(G, G), 0.0)

    def test_positive_for_different_gfs(self):
        from embedding_utils import DOS_diff
        G1 = [np.eye(2, dtype=complex) * complex(0, -0.5)]
        G2 = [np.eye(2, dtype=complex) * complex(0, -1.0)]
        diff = DOS_diff(G1, G2)
        assert diff > 0.0

    def test_unequal_length_raises(self):
        from embedding_utils import DOS_diff
        G1 = [np.eye(2, dtype=complex)]
        G2 = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
        with pytest.raises(ValueError):
            DOS_diff(G1, G2)

    def test_symmetry(self):
        """DOS_diff(G1, G2) == DOS_diff(G2, G1)."""
        from embedding_utils import DOS_diff
        G1 = [np.eye(2, dtype=complex) * complex(0, -0.5)]
        G2 = [np.eye(2, dtype=complex) * complex(0, -1.0)]
        assert np.isclose(DOS_diff(G1, G2), DOS_diff(G2, G1))


# ---------------------------------------------------------------------------
# lanczos.py — Lanczos-to-SOP conversion
# ---------------------------------------------------------------------------

class TestLanczosToSOP:

    def test_empty_input(self):
        from lanczos import lanczos_to_SOP
        res, pol = lanczos_to_SOP([], [], [])
        assert res == []
        assert pol == []

    def test_single_element(self):
        """Single α gives a single pole equal to α."""
        from lanczos import lanczos_to_SOP
        alpha = [2.0]
        beta = [1.0]
        gamma = [1.0]
        res, pol = lanczos_to_SOP(alpha, beta, gamma)
        assert len(pol) == 1
        assert np.isclose(pol[0].real, 2.0, atol=1e-10)

    def test_two_element_count(self):
        """Two Lanczos iterations give two poles."""
        from lanczos import lanczos_to_SOP
        alpha = [-1.0, 1.0]
        beta = [1.0, 0.5]
        gamma = [1.0, 0.5]
        res, pol = lanczos_to_SOP(alpha, beta, gamma)
        assert len(pol) == 2
        assert len(res) == 2

    def test_residues_sum_to_one_for_normalized_input(self):
        """For a normalized single-pole case, residue is the only weight."""
        from lanczos import lanczos_to_SOP
        alpha = [3.0]
        beta = [1.0]
        gamma = [1.0]
        res, pol = lanczos_to_SOP(alpha, beta, gamma)
        # The single residue should be 1 (projection onto first Krylov vector)
        assert np.isclose(res[0], 1.0, atol=1e-10)


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestLanczosBasis:
    """Test that the Lanczos basis tridiagonalizes the Hamiltonian."""

    def test_lanczos_basis_diagonal_matrix(self):
        """Starting from the first basis vector of a diagonal H, α = H[0,0]."""
        from lanczos import lanczos_basis
        from scipy.sparse import csc_matrix
        H = csc_matrix(np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex))
        phi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        a_list, b_list = lanczos_basis(phi0, H, dim=1)
        assert np.isclose(a_list[0], 1.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
