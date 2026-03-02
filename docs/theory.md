# Theoretical Background

## Dynamical Mean Field Theory (DMFT)

DMFT is a non-perturbative approximation for solving strongly correlated electron systems. It maps the original lattice problem onto a single-site impurity problem embedded in a self-consistently determined bath.

### Key Equations

The fundamental DMFT equations are:

1. **Local Green's function:**
   ```
   G_loc(ωn) = (1/Nk) Σ_k [ωn + μ - ε_k - Σ(ωn)]^(-1)
   ```

2. **Self-consistency condition:**
   The impurity Green's function must equal the local (lattice) Green's function:
   ```
   G_imp(ωn) = G_loc(ωn), Sigma_imp(ωn) = Sigma_loc(ωn)
   ```

3. **Embedding potential (hybridization function):**
   ```
   v_emb(ωn) = ωn + μ - h_A- (G_loc^(-1)(ωn) + Σ_A(ωn))
   ```

### DMFT Loop

1. Initialize local Green's function or self-energy (and chemical potential)
2. Calculate embedding potential from the previous quantity, building the Weiss field
3. Solve the quantum impurity problem
4. Extract self-energy from impurity solver
5. Calculate local Green's function
6. Check convergence; if not converged, go to step 2

## Many-body AIM-SOP Hamiltonian (AIM)

The many-body AIM-SOP maps the lattice problem into a quantum impurity problem. It represents the bath degrees of freedom explicitly through a set of fictitious bath sites in an auxiliary Hamiltonian

### Auxiliary Hamiltonian

```
H_AIM = H_A + Σ_i Gamma_i (c^†d_i + h.c.) + Σ_i sigma_i d^†_i d_i
```

Where:
- c: impurity electron operator
- d_i: bath site operators
- Gamma_i: residue of the SOP representation (coupling fragment-auxiliary bath)
- sigma_i: pole of the SOP representation (bath single-particle energy)

## References

- A. Georges, G. Kotliar, W. Krauth, M. J. Rozenberg, "Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions", *Rev. Mod. Phys.* **68**, 13 (1996)

- A. Carbone, M. Capone, N. Marzari, and T. Chiarotti, in preparation (2026)

---

For more details on the implementation, see the inline code documentation.
