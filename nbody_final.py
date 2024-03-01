import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

G = 6.67430e-11  # gravitational constant
c = 299792458  # speed of light
AU = 1.495978707e11  # astronomical unit
MSun = 1.98847e30  # solar mass
yr = 31556952  # year

# function which returns derivative at given time
def func_r_v_dot(r_v,t,N_obj,m): # r_v (positions and velocities) and m (masses) should numpy arrays
    r_v = np.array(r_v) # shape (N_obj*6)
    m = np.array(m)

    r_v_dot = np.zeros((N_obj,6))

    assert r_v.shape == (N_obj*6,)
    assert m.shape == (N_obj,)

    r_v = np.reshape(r_v,(N_obj,6)) # shape N_obj X 6

    for i in range(N_obj):
        r_i = r_v[i,0:3]
        v_i = r_v[i,3:6]
        m_i = m[i]
        a_i = np.zeros(3)
        for j in range(N_obj):
            if i!=j:
                r_j = r_v[j,0:3]
                v_j = r_v[j,3:6]
                m_j = m[j]
                r_ij = r_i-r_j
                r_ij_norm = np.linalg.norm(r_ij)
                a_ij_0 = -G*m_j*r_ij/r_ij_norm**3 # Newtonian acceleration
                a_ij_1 = 0 # 1PN term
                a_ij_25 = 0 # 2.5PN term
                a_i = a_i + a_ij_0 + a_ij_1/c**2 + a_ij_25/c**5
        r_v_dot[i,0:3] = v_i
        r_v_dot[i,3:6] = a_i

    r_v_dot = np.reshape(r_v_dot,(N_obj*6)) # shape (N_obj*6)
    return r_v_dot

# function which returns adaptive time-steps
def func_adapt_dt(r_v,N_obj,m): # r_v (positions and velocities) and m (masses) should numpy arrays
    r_v = np.array(r_v) # shape (N_obj*6)
    m = np.array(m)
    eps = 0

    dt = yr # standard time-step

    assert r_v.shape == (N_obj*6,)
    assert m.shape == (N_obj,)

    r_v = np.reshape(r_v,(N_obj,6)) # shape N_obj X 6

    for i in range(N_obj):
        r_i = r_v[i,0:3]
        v_i = r_v[i,3:6]
        m_i = m[i]
        for j in range(N_obj):
            if i!=j:
                r_j = r_v[j,0:3]
                v_j = r_v[j,3:6]
                m_j = m[j]
                r_ij = r_i-r_j+eps                
                v_ij = v_i-v_j
                r_ij_norm = np.linalg.norm(r_ij)
                v_ij_norm = np.linalg.norm(v_ij)
                a_ij_norm = G*m_j/r_ij_norm**2
                if v_ij_norm == 0 and a_ij_norm != 0:
                    dt_min = np.sqrt(r_ij_norm/a_ij_norm)/4
                elif v_ij_norm != 0 and a_ij_norm == 0:
                    dt_min = (r_ij_norm/v_ij_norm)/4
                elif v_ij_norm != 0 and a_ij_norm != 0:
                    dt_min = min([(r_ij_norm/v_ij_norm)/4,np.sqrt(r_ij_norm/a_ij_norm)/4]) # approximate timescale which needs to be resolved
                else:
                    dt_min = dt
                if dt_min < dt:
                    dt = dt_min
    return dt


def compute_unit_orbit_vecs_from_angles(inc,LAN,AP):
    # major unit vector, minor unit vector and normal unit vector
    e_uvec = np.array([ np.cos(LAN)*np.cos(AP) - np.sin(LAN)*np.sin(AP)*np.cos(inc) , np.sin(LAN)*np.cos(AP) + np.cos(LAN)*np.sin(AP)*np.cos(inc) , np.sin(AP)*np.sin(inc) ])
    q_uvec = np.array([ - np.cos(LAN)*np.sin(AP) - np.sin(LAN)*np.cos(AP)*np.cos(inc) , - np.sin(LAN)*np.sin(AP) + np.cos(LAN)*np.cos(AP)*np.cos(inc) , np.cos(AP)*np.sin(inc) ])
    h_uvec = np.array([ np.sin(LAN)*np.sin(inc) , - np.cos(LAN)*np.sin(inc) , np.cos(inc) ])
    
    return e_uvec,q_uvec,h_uvec

def compute_rel_pos_vel_from_orbit_vecs(e_uvec,q_uvec,a,e,m1,m2,phase):
    assert e_uvec.shape == (3,) and q_uvec.shape == (3,)
    # relative position and velocity vectors
    r_vec = a*(1-e**2) / (1+e*np.cos(phase)) * (np.cos(phase)*e_uvec + np.sin(phase)*q_uvec)
    v_vec = np.sqrt( G*(m1+m2) / (a*(1-e**2)) ) * (-np.sin(phase)*e_uvec + (e+np.cos(phase))*q_uvec)
    
    return r_vec,v_vec


# function which returns initial positions and velocities of a binary
def init_bin(masses,smaxes,eccs,incs,LANs,APs,tAnos):
    assert masses.shape == (2,)
    assert smaxes.shape == (1,)
    assert eccs.shape == (1,)
    assert incs.shape == (1,)
    assert LANs.shape == (1,)
    assert APs.shape == (1,)
    assert tAnos.shape == (1,)

    e_unit = np.zeros([1,3])
    q_unit = np.zeros([1,3])
    h_unit = np.zeros([1,3])

    rrel_0 = np.zeros([1,3])
    vrel_0 = np.zeros([1,3])

    r_v_0 = np.zeros(12)

    # unit vectors of each orbit in the directions of major axis, minor axis and normal
    e_unit[0],q_unit[0],h_unit[0] = compute_unit_orbit_vecs_from_angles(incs[0],LANs[0],APs[0])

    # relative position and velocity vectors
    rrel_0[0],vrel_0[0] = compute_rel_pos_vel_from_orbit_vecs(e_unit[0],q_unit[0],smaxes[0],eccs[0],masses[0],masses[1],tAnos[0])

    m1,m2 = masses
    mtot = np.sum(masses)

    # individual positions and velocities
    r_v_0[0:3] = rrel_0[0] * m2 / (m1+m2)
    r_v_0[3:6] = vrel_0[0] * m2 / (m1+m2)
    r_v_0[6:9] = - rrel_0[0] * m1 / (m1+m2)
    r_v_0[9:12] = - vrel_0[0] * m1 / (m1+m2)

    return r_v_0

# function which returns initial positions and velocities of a triple
def init_trip(masses,smaxes,eccs,incs,LANs,APs,tAnos):
    assert masses.shape == (3,)
    assert smaxes.shape == (2,)
    assert eccs.shape == (2,)
    assert incs.shape == (2,)
    assert LANs.shape == (2,)
    assert APs.shape == (2,)
    assert tAnos.shape == (2,)

    e_unit = np.zeros([2,3])
    q_unit = np.zeros([2,3])
    h_unit = np.zeros([2,3])

    rrel_0 = np.zeros([2,3])
    vrel_0 = np.zeros([2,3])

    r_v_0 = np.zeros(18)

    # unit vectors of each orbit in the directions of major axis, minor axis and normal
    for i in range(2):
        e_unit[i],q_unit[i],h_unit[i] = compute_unit_orbit_vecs_from_angles(incs[i],LANs[i],APs[i])

    # relative position and velocity vectors
    rrel_0[0],vrel_0[0] = compute_rel_pos_vel_from_orbit_vecs(e_unit[0],q_unit[0],smaxes[0],eccs[0],masses[0],masses[1],tAnos[0])
    rrel_0[1],vrel_0[1] = compute_rel_pos_vel_from_orbit_vecs(e_unit[1],q_unit[1],smaxes[1],eccs[1],masses[0]+masses[1],masses[2],tAnos[1])

    m1,m2,m3 = masses
    mtot = np.sum(masses)
    # center of mass of the inner binary (triple center of mass assumed to have 0 position and velocity)
    rCOM_12 = rrel_0[1] * m3 / mtot
    vCOM_12 = vrel_0[1] * m3 / mtot

    # individual positions and velocities
    r_v_0[0:3] = rCOM_12 + rrel_0[0] * m2 / (m1+m2)
    r_v_0[3:6] = vCOM_12 + vrel_0[0] * m2 / (m1+m2)
    r_v_0[6:9] = rCOM_12 - rrel_0[0] * m1 / (m1+m2)
    r_v_0[9:12] = vCOM_12 - vrel_0[0] * m1 / (m1+m2)
    r_v_0[12:15] = - rrel_0[1] * (m1+m2) / mtot
    r_v_0[15:18] = - vrel_0[1] * (m1+m2) / mtot

    return r_v_0

# function which returns initial positions and velocities of a 2+2 quadruple
def init_2p2quad(masses,smaxes,eccs,incs,LANs,APs,tAnos):
    assert masses.shape == (4,)
    assert smaxes.shape == (3,)
    assert eccs.shape == (3,)
    assert incs.shape == (3,)
    assert LANs.shape == (3,)
    assert APs.shape == (3,)
    assert tAnos.shape == (3,)

    e_unit = np.zeros([3,3])
    q_unit = np.zeros([3,3])
    h_unit = np.zeros([3,3])

    rrel_0 = np.zeros([3,3])
    vrel_0 = np.zeros([3,3])

    r_v_0 = np.zeros(24)

    # unit vectors of each orbit in the directions of major axis, minor axis and normal
    for i in range(3):
        e_unit[i],q_unit[i],h_unit[i] = compute_unit_orbit_vecs_from_angles(incs[i],LANs[i],APs[i])

    # relative position and velocity vectors
    rrel_0[0],vrel_0[0] = compute_rel_pos_vel_from_orbit_vecs(e_unit[0],q_unit[0],smaxes[0],eccs[0],masses[0],masses[1],tAnos[0])
    rrel_0[1],vrel_0[1] = compute_rel_pos_vel_from_orbit_vecs(e_unit[1],q_unit[1],smaxes[1],eccs[1],masses[2],masses[3],tAnos[1])
    rrel_0[2],vrel_0[2] = compute_rel_pos_vel_from_orbit_vecs(e_unit[2],q_unit[2],smaxes[2],eccs[2],masses[0]+masses[1],masses[2]+masses[3],tAnos[2])

    m1,m2,m3,m4 = masses
    mtot = np.sum(masses)
    # centers of masses of the two inner binaries (quadruple center of mass assumed to have 0 position and velocity)
    rCOM_12 = rrel_0[2] * (m3+m4) / mtot
    vCOM_12 = vrel_0[2] * (m3+m4) / mtot
    rCOM_34 = - rrel_0[2] * (m1+m2) / mtot
    vCOM_34 = - vrel_0[2] * (m1+m2) / mtot

    # individual positions and velocities
    r_v_0[0:3] = rCOM_12 + rrel_0[0] * m2 / (m1+m2)
    r_v_0[3:6] = vCOM_12 + vrel_0[0] * m2 / (m1+m2)
    r_v_0[6:9] = rCOM_12 - rrel_0[0] * m1 / (m1+m2)
    r_v_0[9:12] = vCOM_12 - vrel_0[0] * m1 / (m1+m2)
    r_v_0[12:15] = rCOM_34 + rrel_0[1] * m4 / (m3+m4)
    r_v_0[15:18] = vCOM_34 + vrel_0[1] * m4 / (m3+m4)
    r_v_0[18:21] = rCOM_34 - rrel_0[1] * m3 / (m3+m4)
    r_v_0[21:24] = vCOM_34 - vrel_0[1] * m3 / (m3+m4)

    return r_v_0

# function which returns initial positions and velocities of a 3+1 quadruple
def init_3p1quad(masses,smaxes,eccs,incs,LANs,APs,tAnos):
    assert masses.shape == (4,)
    assert smaxes.shape == (3,)
    assert eccs.shape == (3,)
    assert incs.shape == (3,)
    assert LANs.shape == (3,)
    assert APs.shape == (3,)
    assert tAnos.shape == (3,)

    e_unit = np.zeros([3,3])
    q_unit = np.zeros([3,3])
    h_unit = np.zeros([3,3])

    rrel_0 = np.zeros([3,3])
    vrel_0 = np.zeros([3,3])

    r_v_0 = np.zeros(24)

    # unit vectors of each orbit in the directions of major axis, minor axis and normal
    for i in range(3):
        e_unit[i],q_unit[i],h_unit[i] = compute_unit_orbit_vecs_from_angles(incs[i],LANs[i],APs[i])

    # relative position and velocity vectors
    rrel_0[0],vrel_0[0] = compute_rel_pos_vel_from_orbit_vecs(e_unit[0],q_unit[0],smaxes[0],eccs[0],masses[0],masses[1],tAnos[0])
    rrel_0[1],vrel_0[1] = compute_rel_pos_vel_from_orbit_vecs(e_unit[1],q_unit[1],smaxes[1],eccs[1],masses[0]+masses[1],masses[2],tAnos[1])
    rrel_0[2],vrel_0[2] = compute_rel_pos_vel_from_orbit_vecs(e_unit[2],q_unit[2],smaxes[2],eccs[2],masses[0]+masses[1]+masses[2],masses[3],tAnos[2])

    m1,m2,m3,m4 = masses
    mtot = np.sum(masses)
    # center of mass of the intermediate binary (quadruple center of mass assumed to have 0 position and velocity)
    rCOM_123 = rrel_0[2] * m4 / mtot
    vCOM_123 = vrel_0[2] * m4 / mtot
    # center of mass of the inner binary
    rCOM_12 = rCOM_123 + rrel_0[1] * m3 / (m1+m2+m3)
    vCOM_12 = vCOM_123 + vrel_0[1] * m3 / (m1+m2+m3)

    # individual positions and velocities
    r_v_0[0:3] = rCOM_12 + rrel_0[0] * m2 / (m1+m2)
    r_v_0[3:6] = vCOM_12 + vrel_0[0] * m2 / (m1+m2)
    r_v_0[6:9] = rCOM_12 - rrel_0[0] * m1 / (m1+m2)
    r_v_0[9:12] = vCOM_12 - vrel_0[0] * m1 / (m1+m2)
    r_v_0[12:15] = rCOM_123 - rrel_0[1] * (m1+m2) / (m1+m2+m3)
    r_v_0[15:18] = vCOM_123 - vrel_0[1] * (m1+m2) / (m1+m2+m3)
    r_v_0[18:21] = - rrel_0[2] * (m1+m2+m3) / mtot
    r_v_0[21:24] = - vrel_0[2] * (m1+m2+m3) / mtot

    return r_v_0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-C","--configuration",help="configuration - bin, trip, 2p2quad, 3p1quad",default="trip")
    parser.add_argument("-m","--masses",help="list of masses in MSun",default=None)
    parser.add_argument("-a","--smaxes",help="list of semimajor axes in AU",default=None)
    parser.add_argument("-e","--eccs",help="list of eccentricities (between 0 and 1)",default=None)
    parser.add_argument("-i","--incs",help="list of inclinations in deg (between 0 and 180)",default=None)
    parser.add_argument("-o","--LANs",help="list of longitudes of ascending node in deg (between 0 and 360)",default=None)
    parser.add_argument("-w","--APs",help="list of arguments of periapsis in deg (between 0 and 360)",default=None)
    parser.add_argument("-t","--tAnos",help="list of true anomalies in deg (between 0 and 360)",default=None)
    parser.add_argument("-N","--N_steps",help="number of time steps",default=1000,type=int)
    parser.add_argument("-F","--fade_factor",help="rate of fading trajectory trails (0 is no fade, < 0.05 for best results)",default=0.01,type=float)
    parser.add_argument("-L","--light_mode",help="aimation in light mode (default dark)",action="store_true")
    parser.add_argument("-S","--save_gif",help="save GIF or not",action="store_true")
    args = parser.parse_args()

    configuration = args.configuration

    masses = eval(args.masses) if args.masses is not None else None
    smaxes = eval(args.smaxes) if args.smaxes is not None else None
    eccs = eval(args.eccs) if args.eccs is not None else None
    incs = eval(args.incs) if args.incs is not None else None
    LANs = eval(args.LANs) if args.LANs is not None else None
    APs = eval(args.APs) if args.APs is not None else None
    tAnos = eval(args.tAnos) if args.tAnos is not None else None

    if configuration == "bin":
        if masses is None:
            masses = [10,30]
        if smaxes is None:
            smaxes = [80]
        if eccs is None:
            eccs = [0.25]
        if incs is None:
            incs = [60]
        if LANs is None:
            LANs = [0]
        if APs is None:
            APs = [0]
        if tAnos is None:
            tAnos = [0]

        N_obj = 2
        m = np.array(masses)*MSun
        min_xyz = - 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        max_xyz = 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        gifsave = "bin_%d_%d_%d_%.2f.gif" \
            %(masses[0],masses[1],smaxes[0],eccs[0],)
        
        r_v_0 = init_bin(np.array(masses)*MSun,np.array(smaxes)*AU,np.array(eccs), np.array(incs)*np.pi/180, \
                             np.array(LANs)*np.pi/180,np.array(APs)*np.pi/180,np.array(tAnos)*np.pi/180)

    elif configuration == "trip":
        if masses is None:
            masses = [30,10,20]
        if smaxes is None:
            smaxes = [15,100]
        if eccs is None:
            eccs = [0.25,0.15]
        if incs is None:
            incs = [60,30]
        if LANs is None:
            LANs = [0,0]
        if APs is None:
            APs = [0,0]
        if tAnos is None:
            tAnos = [0,0]

        N_obj = 3
        m = np.array(masses)*MSun
        min_xyz = - 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        max_xyz = 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        gifsave = "trip_%d_%d_%d_%d_%d_%.2f_%.2f.gif" \
            %(masses[0],masses[1],masses[2],smaxes[0],smaxes[1],eccs[0],eccs[1])
        
        r_v_0 = init_trip(np.array(masses)*MSun,np.array(smaxes)*AU,np.array(eccs), np.array(incs)*np.pi/180, \
                             np.array(LANs)*np.pi/180,np.array(APs)*np.pi/180,np.array(tAnos)*np.pi/180)

    elif configuration == "2p2quad":
        if masses is None:
            masses = [40,20,30,10]
        if smaxes is None:
            smaxes = [10,10,100]
        if eccs is None:
            eccs = [0.25,0.25,0.15]
        if incs is None:
            incs = [30,60,0]
        if LANs is None:
            LANs = [0,0,0]
        if APs is None:
            APs = [0,0,0]
        if tAnos is None:
            tAnos = [0,0,0]

        N_obj = 4
        m = np.array(masses)*MSun
        min_xyz = - 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        max_xyz = 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        gifsave = "2p2_quad_%d_%d_%d_%d_%d_%d_%d_%.2f_%.2f_%.2f.gif" \
            %(masses[0],masses[1],masses[2],masses[3],smaxes[0],smaxes[1],smaxes[2],eccs[0],eccs[1],eccs[2])

        r_v_0 = init_2p2quad(np.array(masses)*MSun,np.array(smaxes)*AU,np.array(eccs), np.array(incs)*np.pi/180, \
                             np.array(LANs)*np.pi/180,np.array(APs)*np.pi/180,np.array(tAnos)*np.pi/180)

    elif configuration == "3p1quad":
        if masses is None:
            masses = [40,10,30,20]
        if smaxes is None:
            smaxes = [20,80,350]
        if eccs is None:
            eccs = [0.25,0.15,0.1]
        if incs is None:
            incs = [60,30,0]
        if LANs is None:
            LANs = [0,0,0]
        if APs is None:
            APs = [0,0,0]
        if tAnos is None:
            tAnos = [0,0,0]

        N_obj = 4
        m = np.array(masses)*MSun
        min_xyz = - 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        max_xyz = 1.2 * (smaxes[-1]/2) * (1 + eccs[-1])
        gifsave = "3p1_quad_%d_%d_%d_%d_%d_%d_%d_%.2f_%.2f_%.2f.gif" \
            %(masses[0],masses[1],masses[2],masses[3],smaxes[0],smaxes[1],smaxes[2],eccs[0],eccs[1],eccs[2])
        
        r_v_0 = init_3p1quad(np.array(masses)*MSun,np.array(smaxes)*AU,np.array(eccs), np.array(incs)*np.pi/180, \
                             np.array(LANs)*np.pi/180,np.array(APs)*np.pi/180,np.array(tAnos)*np.pi/180)

    else:
        print("Wrong input")
        exit()


    # setting up for actual integration and eventual animation    
    N_steps = args.N_steps

    t = np.zeros(N_steps+1)
    r_v_total = np.zeros((N_steps+1,N_obj,6)) # stores positions and velocities after each time-step
    r_v_total[0,:,:] = np.reshape(r_v_0,(N_obj,6))

    r_v_evolve = r_v_0 # position and velocity at a given evolution time
    for step in range(N_steps):
        dt_evolve = func_adapt_dt(r_v_evolve,N_obj,m)
        r_v_evolve = odeint(func_r_v_dot,r_v_evolve,[0,dt_evolve],args=(N_obj,m))[1] # shape 1 X (N_obj*6) -- (0th element is not relevant)

        t[step+1] = t[step]+dt_evolve
        r_v_total[step+1,:,:] = np.reshape(r_v_evolve,(N_obj,6))

    print(t[-1]/yr,"yr evolution time")

    if args.light_mode:
        plt.style.use('default')
        colors = ['mediumvioletred','darkgoldenrod','mediumseagreen','royalblue']
    else:
        plt.style.use('dark_background')
        colors = ['hotpink','gold','chartreuse','deepskyblue']

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')

    trajectories = [LineCollection([],lw=2,color=colors[i]) for i in range(N_obj)]
    for i in range(N_obj):
        ax.add_collection3d(trajectories[i])
    particles = [ax.plot([],[],[],'o',markersize=np.sqrt(m[i]/sum(m))*20,color=colors[i])[0] for i in range(N_obj)]
    
    fade_factor = args.fade_factor
    def animate(t_step,N_obj,r_v,trajectories,particles,t):
        for i in range(N_obj):
            r = r_v[0:t_step+1,i,0:3]/AU # shape * X 3

            # plot each line segment individually with adjusted alpha
            segments = []
            alphas = []
            for step in range(t_step):            
                r1 = r[step:step+2,0]
                r2 = r[step:step+2,1]
                r3 = r[step:step+2,2]
                segments.append( list(zip(r1,r2,r3)) )
                alphas.append( max(0, 1.0 - fade_factor * (t_step - step)) ) # gradual change in alpha
            alphas.append(1.0)
            trajectories[i].set_segments(segments)
            trajectories[i].set_alpha(alphas)

            particles[i].set_data(r[t_step:t_step+1,0],r[t_step:t_step+1,1]) # X and Y axes
            particles[i].set_3d_properties(r[t_step:t_step+1,2]) # Z axis

    # animate N_steps frames with interval 1ms between frames
    anim = FuncAnimation(fig, animate, frames=N_steps+1, fargs=(N_obj,r_v_total,trajectories,particles,t), interval=1, blit=False)

    # ax.view_init(azim=90, elev=0)
    ax.set_xlim3d([min_xyz,max_xyz])
    ax.set_ylim3d([min_xyz,max_xyz])
    ax.set_zlim3d([min_xyz,max_xyz])

    plt.show()
    if args.save_gif:
        anim.save('%s'%(gifsave))

    # CONVERT GIF TO MP4
    # ffmpeg -r 50 -i anim.gif anim.mp4
