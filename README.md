# Create pretty animations of binaries, triples and quadruples

This is a `python3`-based simple N-body integrator and animator for 2-body, 3-body and 4-body systems. The initial conditions can be supplied in hierarchical coordinates: masses and orbital parameters, including semimajor axes, eccentricities, and orbital angles.

There is one mandatory argument to be specified: `-C` or `--configuration`, which can be one of four configurations - `bin`, `trip`, `2p2quad`, `3p1quad` which stand for binary, triple, 2+2 quadruple and 3+1 quadruple respectively.

The following arguments can be specified to change the initial parameters of a given configuration:
- `-m` or `--masses`
- `-a` or `--smaxes`
- `-e` or `--eccs`
- `-i` or `--incs`
- `-o` or `--LANs`
- `-w` or `--APs`
- `-t` or `--tAnos`

Other arguments control the animation itself, and are as follows:
- `-N` or `--N_steps`, which determines the number of integration time-steps (and thus the length of the animation), with a default of 1000
- `-F` or `--fade_factor`, which determines the rate of fading trajectory trails, with a default of 0.05 (0 is no fade, < 0.05 for best results)
- `-L` or `--light_mode`
- `-S` or `--save_gif`
