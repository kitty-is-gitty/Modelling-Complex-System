import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML



import numpy as np

class experiment:
    """
    object that defines one experiment
    """
    def __init__(self,duration,numff,flfrq,width_flfrq,fldur,rint,coupling_strength,moveparameter,vel):
        """
        initialize the experiment
        """
        self.flfrq     = flfrq 
        self.width_flfrq = width_flfrq
        self.fldur = fldur 
        
        if fldur>(0.5/flfrq):
            raise ValueError("flash duration can be maximum  half of the inverse flash frequency")
        
        self.rint = rint 
        self.coupling_strength = coupling_strength            
        self.vel =  vel
        self.number_fireflies    = numff
        self.duration =  duration
        self.dt = min(self.fldur/15/self.coupling_strength,self.fldur/15.) 
        self.iters = int(duration/self.dt)
        self.moveparameter = moveparameter
        self.flash_counter = np.zeros(self.iters)
        self.max_sim_flashs = np.zeros(self.iters)
        self.r_t = np.zeros(self.iters)
        
        # init fireflies
        self.firefly_list = [] 
        for i in range(self.number_fireflies):
            self.firefly_list.append(firefly(self.flfrq,self.width_flfrq,self.fldur,self.rint,self.coupling_strength,self.moveparameter,self.vel,self.dt))
    
    
    def run(self):
        """
        run the experiment for the chosen setting
        """
        t=0
        while True:
            self.print_status(t)
            self.time_stepper()
            flash_counter_t = self.count_flashs()
            t+=1
            
            # stop the integration if maximum number of iteration was reached or
            # if all fireflies are flashing
            if t>self.iters-1 or max(self.flash_counter)==self.number_fireflies:
                self.max_sim_flashs[t:]=self.number_fireflies
                self.r_t[t:] = 1.
                break
            
            self.flash_counter[t] = flash_counter_t
            self.max_sim_flashs[t] = max(self.flash_counter)
            self.r_t[t] =r_t_i(self.number_fireflies,self.firefly_list)
        
    
    def print_status(self,t):
        """ prints current sync state of the model"""
    
        test_print = t % 1000
        if test_print == 0.:
            print ("Time:" + str(t*self.dt)+ "s, Sync:"+ str(self.max_sim_flashs[t]/self.number_fireflies *100.)+"%")
            
    def time_stepper(self):
        """performs one time step in integration for the model"""
        for i in range(self.number_fireflies):
            # time integration
            self.firefly_list[i].timeIntegration()
            
            # let the fireflies interact
            for j in range(self.number_fireflies):
                self.firefly_list[i].interact(self.firefly_list[j])
                
    def count_flashs(self):
        """ counts the number of flashing fireflies at the current time step"""
        flash_counter =0.
        for i in range(self.number_fireflies):
            flash_counter = flash_counter +  self.firefly_list[i].flash
        
        return flash_counter
        
         
            
class firefly():
    """
    object that defines the properties of one firefly
    """
    def __init__(self,flfrq,width_flfrq,fldur,rint,coupling_strength,moveparameter,vel,dt):
        """
        initialize the firefly properties
        """
        # initialize variables from settings
        self.flash_freq     = np.random.uniform(low=flfrq-width_flfrq,high=flfrq+width_flfrq)  
        self.flash_duration = fldur 
        self.thresold_flash = np.sin(np.pi/2.-self.flash_duration*self.flash_freq*np.pi) 
        self.r_interaction  = rint 
        self.coupling_strength = coupling_strength   
        self.moveparameter = moveparameter
        self.maxabsvel = vel
        self.maxvel = vel
        self.l = 1.0 
        self.dt = dt
        
        # random initialization of other variables
        self.x = np.random.uniform(low=0.,high=self.l)
        self.y = np.random.uniform(low=0.,high=self.l)
        self.intrinsic_omega    = 2*np.pi * self.flash_freq # np.random.uniform(low=0.9,high=1.0)
        self.omega    = self.intrinsic_omega
        self.theta    = np.random.uniform(low=0.,high=2*np.pi)
        self.flash    = flash(self.theta,self.thresold_flash)
        self.uvel     = np.random.uniform(low=-self.maxvel,high=self.maxvel)
        self.vvel     = np.random.uniform(low=-self.maxvel,high=self.maxvel)
    
    def timeIntegration(self):
        """
        integrate the model:
            - move the clock of a firefly
            - move the firefly on the grid (if moveparameter == True)
        """
        self.theta = self.theta + self.dt*self.omega % (2*np.pi)
        self.flash = flash(self.theta,self.thresold_flash)
        if self.moveparameter:        
            self.move()
            self.change_velocity()
        
    def interact(self,other_firefly):
        """
        interaction of a firefly with a other fireflies in its vicinity
        """
        Dx = other_firefly.x - self.x
        Dy =  other_firefly.y - self.y
        r = distance(Dx,Dy)
        
        if r!=0. and r<=self.r_interaction and self.flash==0. and other_firefly.flash==1.:
            self.theta = self.theta + np.sin(other_firefly.theta - self.theta) *self.coupling_strength*self.dt #% (2*np.pi) --> does not work caouse other.theta will also interact again!!! and if modlulo
            #other.theta +=self.K*self.dt
    
    def move(self):
        """
        move a firefly
        """
        dx = self.uvel*self.dt
        dy = self.vvel*self.dt    
        self.x = (self.x +dx) % self.l
        self.y = (self.y +dy) % self.l
        
    def change_velocity(self):
        """
        generate a new velocity vector
        """
        self.uvel     = self.uvel+np.random.uniform(low=-self.maxvel/10.,high=self.maxvel/10.)
        self.vvel     = self.vvel+np.random.uniform(low=-self.maxvel/10.,high=self.maxvel/10.)

# =============================================================================
# Some functions
# =============================================================================
# if a fireflie flashs than return 1.0, if not return 0.0
def flash(theta,thresold_flash):
    if np.sin(theta)>thresold_flash:
        return 1.
    else:
        return 0.
    
#calculate the distance between two points
def distance(x,y):
    return np.sqrt(x**2+y**2)

def r_t_i(nff,Fireflies):
    dummy = 0.
    for i in range(nff):
        dummy =  dummy + np.exp(1.0j * Fireflies[i].theta)
    return 1/float(nff) * np.abs(dummy)

def r_mean(r,T,T_0,iters):
    i_0 = int(T_0 * float(iters)/(T+T_0))
    return np.mean(r[i_0:])


# experiment parameters
duration = 40.       
numff    = 75    

# firefly  parameters
f    =  0.25    
Delta_f = 0.01   
T_flash   =  0.2      
r_int    = 0.5        
K = 0.18   
move = True         
maxvel  = 1.08      

np.random.seed(0)
exp = experiment(duration,
                 numff,
                 f,
                 Delta_f,
                 T_flash,r_int,
                 K,
                 move,
                 maxvel)

# Run the model
exp.run()

# Plot some results
timeArray = np.arange(exp.iters)*exp.dt
plt.figure(1)           
plt.plot(timeArray,exp.flash_counter)
plt.plot(timeArray,exp.max_sim_flashs)
plt.xlabel("time[s]")
plt.ylabel("flashing fireflie")
plt.ylim(0,exp.number_fireflies)
plt.grid()

plt.figure(2)           
plt.plot(timeArray,exp.r_t)
plt.xlabel("time[s]")
plt.ylabel("r_mean")
plt.grid()
plt.ylim(0,1)    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Set random seed for reproducibility
np.random.seed(0)

# Define and initialize experiment parameters
exp = experiment(duration, numff, f, Delta_f, T_flash, r_int, K, move, maxvel)

# Set up the main figure with four subplots in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('whitesmoke')  # Background color for the entire figure

# Scatter plot in (x, y) coordinates (Top-Left)
ax_scatter = axs[0, 0]
ax_scatter.set_facecolor('black')  # Set background color to black
ax_scatter.grid(True, linestyle='--', alpha=0.7)
ax_scatter.set_xlabel("x", fontsize=12, fontweight='bold', color='white')
ax_scatter.set_ylabel("y", fontsize=12, fontweight='bold', color='white')
ax_scatter.set_xlim(0, 1)
ax_scatter.set_ylim(0, 1)
sc = ax_scatter.scatter([], [], s=60, alpha=0.75, edgecolors='k')

# Flash count and r_t plot over time (Top-Right)
ax_time = axs[0, 1]
ax_time.grid(True, linestyle='--', alpha=0.7)
ax_time.set_xlabel("time [s]", fontsize=12, fontweight='bold')
ax_time.set_xlim(0, duration)
ax_time.set_ylim(0, numff)

# Flash count line
line_flash, = ax_time.plot([], [], lw=2, color='royalblue', label="Flash Count")
ax_time.fill_between([], [], color='lightblue', alpha=0.3)
ax_time.set_ylabel("number of flashes", fontsize=12, fontweight='bold', color='royalblue')

# Synchronization measure r_t on a second y-axis
ax2 = ax_time.twinx()
line_rt, = ax2.plot([], [], lw=2, color='darkorange', label="r_t (Sync Measure)")
ax2.set_ylim(0, 2)
ax2.set_ylabel("r_t (Synchronization)", fontsize=12, fontweight='bold', color='darkorange')

# Polar plot for phase synchronization (Bottom-Left)
ax_polar = axs[1, 0]
ax_polar = fig.add_subplot(223, projection='polar')
ax_polar.set_ylim(0, 1)  # r-axis range for the synchronization measure
phase_points, = ax_polar.plot([], [], 'o', color='gray', markersize=8)
sync_line, = ax_polar.plot([], [], color='darkorange', lw=2)  # Line showing r_t radius
arrow = ax_polar.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='darkorange', lw=2))

# Trace line for the arrow's trajectory
trace_line, = ax_polar.plot([], [], color='orange', lw=1, alpha=0.5)

# Phase vs. time plot (Bottom-Right)
ax_phase_time = axs[1, 1]
ax_phase_time.set_xlabel("time [s]", fontsize=12, fontweight='bold')
ax_phase_time.set_ylabel("Phase (radians)", fontsize=12, fontweight='bold')
ax_phase_time.set_xlim(0, duration)
ax_phase_time.set_ylim(0, 30)
ax_phase_time.grid(True, linestyle='--', alpha=0.7)

# Initial lines for each firefly's phase over time
phase_lines = [
    ax_phase_time.plot([], [], lw=1, alpha=0.6, label=f"Firefly {i+1}")[0]
    for i in range(exp.number_fireflies)
]

# Lists to store flash counts, r_t values, and phase trajectories over time
nflashs = []
rt_values = []
phase_trajectories = [[] for _ in range(exp.number_fireflies)]
trace_x = []
trace_y = []

def animate(frame_num):
    """Update function for each frame in the animation."""

    # Update the experiment to the next time step
    exp.time_stepper()

    # Firefly positions and flash states
    x = [firefly.x for firefly in exp.firefly_list]
    y = [firefly.y for firefly in exp.firefly_list]
    colors = ["yellow" if firefly.flash == 1.0 else 'gray' for firefly in exp.firefly_list]
    sizes = [80 if color == "yellow" else 30 for color in colors]

    # Update scatter plot in (x, y)
    sc.set_offsets(np.c_[x, y])
    sc.set_facecolors(colors)
    sc.set_sizes(sizes)

    # Update flash counts and r_t
    nflashs.append(exp.count_flashs())
    rt_value = r_t_i(exp.number_fireflies, exp.firefly_list)
    rt_values.append(rt_value)
    time = np.linspace(0, len(nflashs) * exp.dt, len(nflashs))
    
    # Flash count plot
    line_flash.set_data(time, nflashs)
    ax_time.fill_between(time, nflashs, color='lightblue', alpha=0.3)

    # r_t line plot
    line_rt.set_data(time, rt_values)

    # Calculate average phase for r_t direction in polar plot
    phases = np.array([firefly.theta for firefly in exp.firefly_list])
    avg_phase = np.angle(np.sum(np.exp(1j * phases)) / exp.number_fireflies)  # Mean phase angle
    phase_points.set_data(phases, [1] * len(phases))  # All fireflies placed at r=1
    sync_line.set_data([0, avg_phase], [0, rt_value])  # Sync line with r_t as radius and avg_phase as angle

    # Update arrow position at the tip of r_t line
    arrow.xy = (avg_phase, rt_value)

    # Store current polar coordinates for the trace
    trace_x.append(avg_phase)
    trace_y.append(rt_value)

    # Update trace line to show the trajectory
    trace_line.set_data(trace_x, trace_y)

    # Update phase vs. time plot for each firefly
    for i, (firefly, line) in enumerate(zip(exp.firefly_list, phase_lines)):
        phase_trajectories[i].append(firefly.theta)
        line.set_data(time, phase_trajectories[i])  # Set the phase trajectory data

    # Title for synchronization and flash count
    ax_time.set_title(
        f"Max. simultaneous flashing: {max(nflashs) / exp.number_fireflies * 100:.2f}% | Current r_t: {rt_value:.2f}",
        fontsize=12, fontweight='bold'
    )

# Create and display the animation
ani = animation.FuncAnimation(
    fig, animate, frames=exp.iters, interval=20, repeat=False
)



plt.tight_layout()
plt.show()