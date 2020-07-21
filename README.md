# seminar-qaoa

SS 2020 - Master Seminar - Topics of Quantum Computing - QAOA Knapsack

Our code has 2 main components:

    1. knapsack bruteforce
        A basic 0-1 Knapsack with an approximation ratio of 0.5
    2. qaoa-run
        A comlex function handling the qaoa logic with 2 different optimizer posibilities.
        SLSQP has faster run time becasue it does not need to do a noise removal like COBYLA,
        said it can run noiseless.
        This function also has additional parameters for number of shots, p-level.
    
    For our test runs, we have used a simple arrays of 4 consisting of weights, values. These can be
    customized for higher problems sizes although keep in mind that increasing the maximum weight will
    increase the circuit depth by log(W_max) + 1.