#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

from sirius import DFT_ground_state_find
import sirius.baarman as st
import sirius.ot as ot
from numpy import linspace


def geodesic(X, Y, tau):
    """
    Keyword Arguments:
    X   --
    Y   --
    tau --
    """
    U, _ = st.stiefel_transport_operators(Y, X, tau)
    return U@X


def p(X, Y, tau, E):
    """
    Keyword Arguments:
    X   --
    Y   --
    tau --
    """
    return E(geodesic(X, Y, tau))


def run():
    # run a single SCF iteration to initialize the system
    res = DFT_ground_state_find(1, config='sirius.json')
    # extract wrappers from C++
    density = res['density']
    potential = res['potential']
    kset = res['kpointset']

    # create object to compute the total energy
    E = ot.Energy(kset, potential, density,
                ot.ApplyHamiltonian(potential, kset))
    # get PW coefficients from C++
    X = kset.C
    # get occupation numbers
    fn = kset.fn

    _, HX = E.compute(X)
    dAdC = HX*fn*kset.w
    # project gradient of the free energy to the Stiefel manifold
    Y = st.stiefel_project_tangent(-dAdC, X)
    # evaluate energy along geodesic
    ts = linspace(0, 1.5, 20)
    es = [p(X, Y, t, lambda X: E(X)) for t in ts]

    plt.plot(ts, es, '-x')
    plt.ylabel('Energy [Ha]')
    plt.xlabel(r'$\tau$')
    plt.title(r'Energy along geodesic $X(\tau)$')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run()
