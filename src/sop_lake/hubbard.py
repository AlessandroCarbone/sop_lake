try:
    from qiskit_nature.second_q.operators import FermionicOp
except ImportError:
    FermionicOp = type(None)

def Hubbarb_Ham_1D(t,U,N,bc):
    """Generate single-band (1 state/orbital for each site) 1D Hubbard Hamiltonian inside chain/ring
    t  : Hopping term
    U  : On-site term
    N  : Number of chain sites
    bc : Boundary conditions (open or periodic system)
    """
    # Total number of states inside model Hamiltonian
    ntot = N*2
    # N = num. sites, 2 = degeneracy of each state
    # bc = 0 (open, i.e. chain), bc = 1 (periodic, i.e. ring)
    
    hopping_term, onsite_term = [], []
    
    # Definition of bc inside model
    if N==2:
        end_point = N-1
    else:
        if bc==0:
            end_point = N-1
        else:
            end_point = N 
    
    # Hopping term
    H_hop = 0
    if N!=1:
        coeff = -t
        for i in range(end_point):
            if i==N-1:
                j = 0
            else:
                j = i+1
                
            for k in range(2):
                label1 = "+_"+str(i*2+k)+" -_"+str(j*2+k)
                label2 = "+_"+str(j*2+k)+" -_"+str(i*2+k)
                hopping_term.append({label1: coeff})
                hopping_term.append({label2: coeff}) 
        H_hop = sum(FermionicOp(term,ntot) for term in hopping_term)
    
    # Onsite term
    coeff = U
    for i in range(N):
        label = "+_"+str(i*2)+" -_"+str(i*2)+" +_"+str(i*2+1)+" -_"+str(i*2+1)
        onsite_term.append({label: coeff})
    H_onsite = sum(FermionicOp(term,ntot) for term in onsite_term)
    
    H = H_hop + H_onsite
    return H

def hopping_Ham_1D(t,N,bc):
    """ This function creates the hopping non-interacting Hamiltonian of a 1D chain/ring in the Hubbard model.
    t  : Hopping parameter
    N  : Number of sites of the system
    bc : Boundary conditions (0 for open, 1 for periodic)
    """
    Ntot = N*2
    hopping_term = []
    if N==2:
        end_point = N-1
    else:
        if bc==0:
            end_point = N-1
        else:
            end_point = N 

    coeff = -t
    for i in range(end_point):
        if i==N-1:
            j = 0
        else:
            j = i+1

        for k in range(2):
            label1 = "+_"+str(i*2+k)+" -_"+str(j*2+k)
            label2 = "+_"+str(j*2+k)+" -_"+str(i*2+k)
            hopping_term.append({label1: coeff})
            hopping_term.append({label2: coeff}) 
    h = sum(FermionicOp(term,Ntot) for term in hopping_term)
    return h

def onsite_Ham_1D(U,N):
    Ntot = N*2
    onsite_term = []
    for i in range(N):
        label = "+_"+str(i*2)+" -_"+str(i*2)+" +_"+str(i*2+1)+" -_"+str(i*2+1)
        onsite_term.append({label: U})
    H_onsite = sum(FermionicOp(term,Ntot) for term in onsite_term)
    return H_onsite

def prepare_Hubbard_Hamiltonians(t,U,N,NA,bc=1):
    """ This function prepares the Hubbard Hamiltonians associated with the non-interacting system, the fully-interacting system,
    the non-interacting fragment and the fully-interacting fragment.
    t : Hopping parameter of the Hubbard Hamiltonian
    U : On-site parameter of the Hubbard Hamiltonian
    N : Total number of Hubbard sites in the system
    n : Number of Hubbard sites in the fragment
    bc : Boundary conditions (1 for periodic, 0 for non-periodic)
    """
    if N < 7:  
        H = Hubbarb_Ham_1D(t,U,N,bc)        # True interacting Hamiltonian H
        h = hopping_Ham_1D(t,N,bc)          # Non-interacting Hamiltonian h
    else:
        H, h = None, None                                                 
    HA = Hubbarb_Ham_1D(t,U,NA,bc=0)        # Interacting Hamiltonian H00
    if NA == 1:
        hA = 0.
    else:
        hA = hopping_Ham_1D(t,NA,bc)        # Non-interacting Hamiltonian h00
    return h, H, hA, HA


