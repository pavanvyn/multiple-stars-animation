# Create cool animations of binaries, triples and quadruples

This is a `python3`-based simple N-body integrator and animator for 2-body, 3-body and 4-body systems. The initial conditions can be supplied in hierarchical coordinates: masses and orbital parameters, including semimajor axes, eccentricities, and orbital angles.

There is one (mandatory) argument to be specified: `-C` or `--configuration`, which can be one of four configurations - `bin`, `trip`, `2p2quad`, `3p1quad` which stand for binary (`n_obj`= 2), triple (`n_obj`= 3), 2+2 quadruple (`n_obj`= 4) and 3+1 quadruple (`n_obj`= 4) respectively. If not specified, a triple animation is produced.

The following arguments can be specified to change the initial parameters of a given configuration (it is necessary for these to be lists, even if there is only one element):
- `-m` or `--masses`, a list of length `n_obj` of masses in solar mass units
- `-a` or `--smaxes`, a list of length `n_obj-1` of orbit semimajor axes in astronomical units
- `-e` or `--eccs`, a list of length `n_obj-1` of orbit eccentricities  (between 0 and 1)
- `-i` or `--incs`, a list of length `n_obj-1` of orbit inclinations in degrees (between 0 and 180)
- `-o` or `--LANs`, a list of length `n_obj-1` of orbit longitudes of ascending node in degrees (between 0 and 360)
- `-w` or `--APs`, a list of length `n_obj-1` of orbit arguments of periapsis in degrees (between 0 and 360)
- `-t` or `--tAnos`, a list of length `n_obj-1` of orbit true anomalies in degrees (between 0 and 360)
It is recommended to not specify the true anomalies since they only specify the initial phase in each orbit. If not familiar with the other orbital angles, they need not be specified as well.

Other arguments control the animation itself, and are as follows:
- `-N` or `--N_steps`, the number of integration time-steps (and thus the length of the animation), the default being 1000
- `-F` or `--fade_factor`, the rate of fading trajectory trails, the default being 0.05 (0 is no fade, < 0.05 for best results)
- `-L` or `--light_mode`, to switch the default dark mode to light
- `-S` or `--save_gif`, to save the animation to a gif

Enjoy creating cool animations! A suite of pre-made animations are provided in this repository.
