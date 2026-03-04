"""Integration tests for configuration loading and DMFT simulation setup."""

import pytest
import numpy as np
import tempfile
import yaml

try:
    import qiskit_nature
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# ---------------------------------------------------------------------------
# Minimal valid configuration used across tests
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "system": {
        "size": 100,
        "t": 1.0,
        "U": 1.0,
        "sizeA": 1,
        "Np": 1.0,
        "bc": 1,
    },
    "input": {
        "input_case": "non_int",
    },
    "embedding": {
        "max_iter": 10,
        "num_poles": 4,
        "mu_fixed": 0.5,
        "p_type": "sqrt",
        "axis": "shift",
        "eta_axis": 0.5,
        "num_pts": 100,
        "beta_T": 1500.0,
        "Nw_max": 100,
        "w_edges": [-5.0, 5.0],
        "sparse_gs": True,
        "gs_search": "std",
        "solver_method": "std",
    },
    "optimization": {
        "mixing_method": "linear",
        "alpha": 0.5,
        "opt_method": "scipy_CG",
        "opt_params": [100, 1e-5],
        "initial_mixing": True,
        "complex_poles": False,
        "herm_residues": True,
        "fixed_residues": False,
        "odd_spectrum": False,
        "paramagnetic": True,
        "interp_method": "sampling",
        "print_interp": False,
        "p0_start": "always",
        "thr_diff_prev": 1e-8,
        "thr_stagnation": 1e-9,
        "RMSE_thr": 0.5,
    },
}


# ---------------------------------------------------------------------------
# Hubbard_system_config
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestHubbardSystemConfig:

    def test_basic_initialization(self):
        from src.dmft_config import Hubbard_system_config
        cfg = Hubbard_system_config(size=100, t=1.0, U=1.0, sizeA=1, Np=1.0, bc=1)
        assert cfg.size == 100
        assert cfg.t == 1.0
        assert cfg.U == 1.0
        assert cfg.sizeA == 1
        assert cfg.Np == 1.0

    def test_ntot_is_twice_sizeA(self):
        from src.dmft_config import Hubbard_system_config
        cfg = Hubbard_system_config(size=100, t=1.0, U=1.0, sizeA=2, Np=1.0, bc=1)
        assert cfg.ntot == 4  # 2 * sizeA

    def test_epsk_list_length(self):
        """epsk_list should contain Nk = size/sizeA dispersion values."""
        from src.dmft_config import Hubbard_system_config
        cfg = Hubbard_system_config(size=100, t=1.0, U=1.0, sizeA=1, Np=1.0, bc=1)
        assert len(cfg.epsk_list) == 100

    def test_epsk_dispersion_formula(self):
        """ε_k = -2t cos(2πk/Nk)."""
        from src.dmft_config import Hubbard_system_config
        t = 1.0
        cfg = Hubbard_system_config(size=4, t=t, U=0.0, sizeA=1, Np=1.0, bc=1)
        Nk = 4
        expected = [-2.0 * t * np.cos(2 * np.pi * k / Nk) for k in range(Nk)]
        assert np.allclose(cfg.epsk_list, expected)

    def test_epsk_sum_vanishes(self):
        """For a large ring, Σ_k ε_k = 0 (sum of cosines over full period)."""
        from src.dmft_config import Hubbard_system_config
        cfg = Hubbard_system_config(size=100, t=1.0, U=1.0, sizeA=1, Np=1.0, bc=1)
        assert np.isclose(sum(cfg.epsk_list), 0.0, atol=1e-10)

    def test_epsk_bandwidth(self):
        """For a 1D ring, band edges should be ±2t."""
        from src.dmft_config import Hubbard_system_config
        t = 1.0
        cfg = Hubbard_system_config(size=1000, t=t, U=1.0, sizeA=1, Np=1.0, bc=1)
        assert np.isclose(max(cfg.epsk_list), 2 * t, atol=1e-3)
        assert np.isclose(min(cfg.epsk_list), -2 * t, atol=1e-3)


# ---------------------------------------------------------------------------
# embedding_config
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestEmbeddingConfig:

    def test_default_values(self):
        from src.dmft_config import embedding_config
        cfg = embedding_config()
        assert cfg.max_iter == 300
        assert cfg.num_poles == 4
        assert cfg.p_type == "sqrt"
        assert cfg.axis == "imaginary"

    def test_matsubara_params_set_by_post_init(self):
        from src.dmft_config import embedding_config
        cfg = embedding_config(beta_T=100.0, Nw_max=500)
        assert cfg.matsubara_params["beta"] == 100.0
        assert cfg.matsubara_params["Nw_max"] == 500

    def test_custom_values(self):
        from src.dmft_config import embedding_config
        cfg = embedding_config(num_poles=6, axis="shift", solver_method="lanczos")
        assert cfg.num_poles == 6
        assert cfg.axis == "shift"
        assert cfg.solver_method == "lanczos"


# ---------------------------------------------------------------------------
# optimization_config
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestOptimizationConfig:

    def test_default_alpha(self):
        from src.dmft_config import optimization_config
        cfg = optimization_config()
        assert cfg.alpha == 0.5

    def test_bounds_mirror_flags(self):
        from src.dmft_config import optimization_config
        cfg = optimization_config(
            complex_poles=True,
            herm_residues=False,
            odd_spectrum=True,
            paramagnetic=False,
        )
        assert cfg.bounds["complex_poles"] == True
        assert cfg.bounds["herm_residues"] == False
        assert cfg.bounds["odd_spectrum"] == True
        assert cfg.bounds["paramagnetic"] == False

    def test_convergence_thresholds(self):
        from src.dmft_config import optimization_config
        cfg = optimization_config(thr_diff_prev=1e-6, RMSE_thr=0.1)
        assert cfg.thr_diff_prev == 1e-6
        assert cfg.RMSE_thr == 0.1

    def test_stagnation_default_relative_to_convergence(self):
        """By default thr_stagnation = thr_diff_prev * 0.1."""
        from src.dmft_config import optimization_config
        cfg = optimization_config()
        assert np.isclose(cfg.thr_stagnation, cfg.thr_diff_prev * 0.1)


# ---------------------------------------------------------------------------
# load_sim_config
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestLoadSimConfig:

    def _write_config(self, tmp_path, cfg_dict=None):
        config_file = tmp_path / "config.yaml"
        data = cfg_dict if cfg_dict is not None else SAMPLE_CONFIG
        with open(config_file, "w") as f:
            yaml.dump(data, f)
        return str(config_file)

    def test_returns_sim_config(self, tmp_path):
        from src.dmft_config import load_sim_config, sim_config
        path = self._write_config(tmp_path)
        cfg = load_sim_config(path)
        assert isinstance(cfg, sim_config)

    def test_system_parameters(self, tmp_path):
        from src.dmft_config import load_sim_config
        path = self._write_config(tmp_path)
        cfg = load_sim_config(path)
        assert cfg.system.size == 100
        assert cfg.system.t == 1.0
        assert cfg.system.U == 1.0
        assert cfg.system.sizeA == 1

    def test_embedding_parameters(self, tmp_path):
        from src.dmft_config import load_sim_config
        path = self._write_config(tmp_path)
        cfg = load_sim_config(path)
        assert cfg.embedding.num_poles == 4
        assert cfg.embedding.p_type == "sqrt"
        assert cfg.embedding.axis == "shift"
        assert cfg.embedding.num_pts == 100

    def test_optimization_parameters(self, tmp_path):
        from src.dmft_config import load_sim_config
        path = self._write_config(tmp_path)
        cfg = load_sim_config(path)
        assert cfg.optimization.alpha == 0.5
        assert cfg.optimization.paramagnetic == True
        assert cfg.optimization.opt_method == "scipy_CG"

    def test_float_coercion(self, tmp_path):
        """Numeric optimization fields should be coerced to float."""
        from src.dmft_config import load_sim_config
        data = {k: v for k, v in SAMPLE_CONFIG.items()}
        data["optimization"] = {**SAMPLE_CONFIG["optimization"], "alpha": 1}  # integer
        path = self._write_config(tmp_path, data)
        cfg = load_sim_config(path)
        assert isinstance(cfg.optimization.alpha, float)

    def test_missing_file_raises(self):
        from src.dmft_config import load_sim_config
        with pytest.raises(FileNotFoundError):
            load_sim_config("/nonexistent/path/config.yaml")

    def test_epsk_list_computed(self, tmp_path):
        """After loading, epsk_list should be computed automatically."""
        from src.dmft_config import load_sim_config
        path = self._write_config(tmp_path)
        cfg = load_sim_config(path)
        assert len(cfg.system.epsk_list) == 100  # size / sizeA = 100 / 1


# ---------------------------------------------------------------------------
# sim_config.get_input_variables (non_int case)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit_nature not installed")
class TestGetInputVariables:

    def _make_config(self, tmp_path):
        from src.dmft_config import load_sim_config
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(SAMPLE_CONFIG, f)
        return load_sim_config(str(config_file))

    def test_non_int_returns_five_values(self, tmp_path):
        cfg = self._make_config(tmp_path)
        result = cfg.get_input_variables()
        assert len(result) == 5  # (Gimp_SOP0, Gloc_list0, SigmaA_list0, SOP0, mu0)

    def test_non_int_gimp_is_none(self, tmp_path):
        cfg = self._make_config(tmp_path)
        Gimp_SOP0, _, _, _, _ = cfg.get_input_variables()
        assert Gimp_SOP0 is None

    def test_non_int_mu_is_fixed(self, tmp_path):
        cfg = self._make_config(tmp_path)
        _, _, _, _, mu0 = cfg.get_input_variables()
        assert np.isclose(mu0, 0.5)  # mu_fixed from SAMPLE_CONFIG

    def test_non_int_sigma_zero(self, tmp_path):
        """Starting self-energy should be identically zero."""
        cfg = self._make_config(tmp_path)
        _, _, SigmaA_list0, _, _ = cfg.get_input_variables()
        ntot = cfg.system.ntot
        for S in SigmaA_list0:
            assert np.allclose(S, np.zeros((ntot, ntot), dtype=complex))

    def test_non_int_sop0_poles_count(self, tmp_path):
        """SOP0 should have num_poles poles."""
        cfg = self._make_config(tmp_path)
        _, _, _, SOP0, _ = cfg.get_input_variables()
        assert SOP0.num_poles == cfg.embedding.num_poles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
