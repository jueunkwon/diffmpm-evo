"""
Evolving Composite Soft Robots via Differentiable MPM

This module simulates a differentiable Material Point Method (MPM) for soft robots,
evolves their actuation parameters via an evolutionary algorithm, and records videos 
(before and after optimization) along with saving simulation frames and candidate parameters.
"""

import taichi as ti
import argparse
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import imageio 

# Change the trial number here:
TRIAL_NUM = "trial0012"

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

# -------------------------------------------------------------------
# Simulation Parameters
# -------------------------------------------------------------------
dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0  # will be set in Scene
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

# -------------------------------------------------------------------
# Field Constructors and Global Fields
# -------------------------------------------------------------------
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

# Particle, grid, and simulation fields.
actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

# Boundary and friction coefficient for grid operations
bound = 3
coeff = 0.5

# -------------------------------------------------------------------
# Functions and Kernels
# -------------------------------------------------------------------

def allocate_fields():
    """Allocates and initializes the Taichi fields in the simulation.
    
    This function creates the necessary dense fields on the Taichi root,
    including fields for weights, bias, actuation, particle attributes,
    grid velocities, masses, and loss.
    """
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

@ti.kernel
def clear_grid():
    """Clears the grid by resetting grid velocity and mass fields to zero."""
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0

@ti.kernel
def p2g(f: ti.i32):
    """
    Transfers particle data to grid (Particle-to-Grid step).

    For each particle, computes interpolation weights, updates deformation gradient,
    computes stress and affine velocity field, and accumulates contributions to the grid.
    """
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = new_F.determinant()
        if particle_type[p] == 0:
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])
        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)
        act_id = actuator_id[p]
        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass

@ti.kernel
def grid_op():
    """
    Performs grid operations, including applying gravity and boundary conditions.

    For each grid cell, updates the grid velocity based on mass and applies simple
    collision and friction conditions at the boundaries.
    """
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.0:
                lin = v_out.dot(normal)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff * lin <= 0:
                        v_out[0] = 0
                        v_out[1] = 0
                    else:
                        v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
        grid_v_out[i, j] = v_out

@ti.kernel
def g2p(f: ti.i32):
    """
    Transfers grid data back to particles (Grid-to-Particle step).

    For each particle, interpolates updated grid velocities back to the particle
    and updates the particle's position and affine velocity field.
    """
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * new_v
        C[f + 1, p] = new_C

@ti.kernel
def compute_actuation(t: ti.i32):
    """
    Computes the actuation for each actuator at time step t.

    This function calculates a weighted sum of sinusoidal functions (plus bias)
    for each actuator and applies a hyperbolic tangent to generate an actuation signal.
    """
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt + 2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)

@ti.kernel
def compute_x_avg():
    """
    Computes the average x-coordinate of all solid particles.
    
    The contribution of each solid particle is added to x_avg.
    """
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])

@ti.kernel
def compute_loss():
    """
    Computes the loss value for the simulation.

    The loss is defined as the negative average x-coordinate of the solid particles.
    """
    dist = x_avg[None][0]
    loss[None] = -dist

def forward(total_steps=steps, record=False, record_interval=1):
    """
    Runs the simulation forward for a specified number of steps.

    Optionally records frames at intervals if 'record' is True.
    
    Args:
        total_steps (int): Total number of time steps to run.
        record (bool): Whether to record frames.
        record_interval (int): Interval between recorded frames.
    
    Returns:
        list: A list of recorded frames as uint8 images if record is True.
    """
    frames = []
    for s in range(total_steps - 1):
        clear_grid()
        compute_actuation(s)
        p2g(s)
        grid_op()
        g2p(s)
        if record and s % record_interval == 0:
            gui.clear(0xFFFFFF)
            # Use particle positions from step s+1
            particles = x.to_numpy()[s+1]
            aid = actuator_id.to_numpy()
            act_array = actuation.to_numpy()
            colors = np.empty(n_particles, dtype=np.uint32)
            for i in range(n_particles):
                color = 0x111111
                if aid[i] != -1:
                    act = act_array[s, int(aid[i])]
                    color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
                colors[i] = color
            gui.circles(pos=particles, color=colors, radius=1.5)
            gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
            # Rotate frame 90° counterclockwise to get the floor at the bottom.
            frame = gui.get_image()
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_corrected = np.rot90(frame_uint8, k=1)
            frames.append(frame_corrected)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()
    if record:
        return frames

def visualize(frame_idx, folder):
    """
    Saves an individual frame as an image to a specified folder.
    
    Args:
        frame_idx (int): The simulation step whose frame is saved.
        folder (str): The destination folder for the image.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    particles = x.to_numpy()[frame_idx]
    aid = actuator_id.to_numpy()
    act_array = actuation.to_numpy()
    colors = np.empty(n_particles, dtype=np.uint32)
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = act_array[frame_idx - 1, int(aid[i])] if frame_idx > 0 else 0.0
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.clear(0xFFFFFF)
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    file_path = os.path.join(folder, f"frame_{frame_idx:04d}.png")
    gui.show(file_path)

def save_visualization(gen):
    """
    Saves a final visualization image (at the end of simulation) for a given generation.
    
    Args:
        gen (int): The generation number used to label the saved image.
    """
    particles = x.to_numpy()[steps - 1]
    aid = actuator_id.to_numpy()
    act_array = actuation.to_numpy()
    colors = np.empty(n_particles, dtype=np.uint32)
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = act_array[steps - 2, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    file_path = os.path.join(TRIAL_NUM, f"{TRIAL_NUM}_{gen:03d}_.png")
    gui.show(file_path)

# -------------------------------------------------------------------
# Scene Class for Composite Soft Robot Generation
# -------------------------------------------------------------------
class Scene:
    """
    A class to procedurally generate composite soft robot structures.

    Attributes:
        n_particles (int): Total number of particles.
        n_solid_particles (int): Total number of solid particles.
        x (list): List of particle positions.
        actuator_id (list): List of actuator IDs for particles.
        particle_type (list): List indicating the type of each particle.
        offset_x (float): Horizontal offset for the structure.
        offset_y (float): Vertical offset for the structure.
    """
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []             # List of particle positions
        self.actuator_id = []   # List of actuator ids
        self.particle_type = [] # List indicating type (e.g., fluid vs. solid)
        self.offset_x = 0
        self.offset_y = 0

    def add_circle(self, cx, cy, radius, actuation, ptype=1):
        """
        Adds a circular region of particles to the structure.

        Args:
            cx (float): Center x-coordinate.
            cy (float): Center y-coordinate.
            radius (float): Radius of the circle.
            actuation (int): Actuator ID for the particles (-1 for none).
            ptype (int): Particle type (1 for solid, 0 for fluid).
        """
        if ptype == 0:
            assert actuation == -1
        global n_particles
        num_points_x = int(10 * radius / dx)
        num_points_y = int(10 * radius / dx)
        for i in range(num_points_x):
            for j in range(num_points_y):
                grid_x = cx + (i - num_points_x // 2) * dx
                grid_y = cy + (j - num_points_y // 2) * dx
                if (grid_x - cx)**2 + (grid_y - cy)**2 <= radius**2:
                    self.x.append([grid_x + self.offset_x, grid_y + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)

    def add_rect(self, cx, cy, w, h, actuation, ptype=1):
        """
        Adds a rectangular region of particles to the structure.

        Args:
            cx (float): Center x-coordinate.
            cy (float): Center y-coordinate.
            w (float): Width of the rectangle.
            h (float): Height of the rectangle.
            actuation (int): Actuator ID for the particles (-1 for none).
            ptype (int): Particle type.
        """
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    cx + (i + 0.5) * real_dx + self.offset_x,
                    cy + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        """
        Sets the offset for the entire structure.
        
        Args:
            x (float): Horizontal offset.
            y (float): Vertical offset.
        """
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        """
        Finalizes the scene by updating global particle counts.
        """
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        """
        Sets the global number of actuators.

        Args:
            n_act (int): Number of actuators.
        """
        global n_actuators
        n_actuators = n_act

    def build_composite(self, base_shape="rect", num_attachments=4, num_actuators=4):
        """
        Builds a composite structure for a soft robot.

        Args:
            base_shape (str): Shape of the base ('rect' or 'circle').
            num_attachments (int): Number of attachments to add.
            num_actuators (int): Number of actuators.
        """
        self.set_n_actuators(num_actuators)
        base_center = (0.3, 0.15)
        if base_shape == "rect":
            base_width = 0.3
            base_height = 0.05
            base_ll = (base_center[0] - base_width/2, base_center[1] - base_height/2)
            self.add_rect(base_ll[0], base_ll[1], base_width, base_height, actuation=-1, ptype=1)
            base_part = {"type": "rect", "center": base_center, "width": base_width, "height": base_height}
        else:
            base_radius = 0.1
            self.add_circle(base_center[0], base_center[1], base_radius, actuation=-1, ptype=1)
            base_part = {"type": "circle", "center": base_center, "radius": base_radius}
        
        alpha = 0.5  
        offset_range = 0.2  
        for i in range(num_attachments):
            parent = base_part  
            parent_center = parent["center"]
            angle = random.uniform(0, 2 * math.pi)
            if parent["type"] == "rect":
                a = parent["width"] / 2
                b = parent["height"] / 2
                d_exact = float('inf')
                if abs(math.cos(angle)) > 1e-6:
                    d_exact = a / abs(math.cos(angle))
                if abs(math.sin(angle)) > 1e-6:
                    d_exact = min(d_exact, b / abs(math.sin(angle)))
                d_avg = (a + b) / 2
                d_boundary = alpha * d_exact + (1 - alpha) * d_avg
            else:
                d_boundary = parent["radius"]
            
            new_type = random.choice(["circle", "rect"])
            k = random.uniform(-offset_range, offset_range)
            if new_type == "circle":
                att_radius = random.uniform(0.025, 0.06)
                distance = d_boundary + k * att_radius
                distance = max(distance, 0)
                new_center = (parent_center[0] + distance * math.cos(angle),
                              parent_center[1] + distance * math.sin(angle))
                act = random.choice([random.randint(0, num_actuators-1), -1])
                self.add_circle(new_center[0], new_center[1], att_radius, actuation=act, ptype=1)
            else:
                att_width = random.uniform(0.05, 0.1)
                att_height = random.uniform(0.05, 0.1)
                new_size = min(att_width, att_height)
                distance = d_boundary + k * new_size
                distance = max(distance, 0)
                new_center = (parent_center[0] + distance * math.cos(angle),
                              parent_center[1] + distance * math.sin(angle))
                act = random.choice([random.randint(0, num_actuators-1), -1])
                rect_ll = (new_center[0] - att_width/2, new_center[1] - att_height/2)
                self.add_rect(rect_ll[0], rect_ll[1], att_width, att_height, actuation=act, ptype=1)

def composite(scene):
    """
    Builds the composite soft robot structure by invoking the Scene's build_composite method.
    
    Args:
        scene (Scene): The Scene instance to build the structure in.
    """
    scene.build_composite(base_shape="rect", num_attachments=4, num_actuators=4)

gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)

# -------------------------------------------------------------------
# Main Evolutionary Optimization Loop
# -------------------------------------------------------------------
def main():
    """
    Main function for running the evolutionary optimization simulation.

    This function:
      - Parses command-line arguments for generations and population size.
      - Builds the composite soft robot structure.
      - Allocates fields and initializes the simulation.
      - Runs the simulation to record a "before optimization" video.
      - Evolves the candidate parameters for a specified number of generations.
      - Records intermediate visualizations and saves a final "after optimization" video.
      - Appends candidate details (including loss values) to a parameters file.
      - Plots the loss curve over generations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gens', type=int, default=50, help="Number of generations")
    parser.add_argument('--pop', type=int, default=20, help="Population size")
    options = parser.parse_args()

    # Create output folder for the trial.
    if not os.path.exists(TRIAL_NUM):
        os.makedirs(TRIAL_NUM)
    params_file_path = os.path.join(TRIAL_NUM, f"{TRIAL_NUM}_params.txt")

    # Build composite structure.
    scene = Scene()
    composite(scene)
    scene.finalize()
    allocate_fields()

    # Save the initial scene parameters.
    with open(params_file_path, 'w') as f:
        f.write("Initial Scene Parameters:\n")
        f.write("Number of particles: {}\n".format(scene.n_particles))
        f.write("Particle positions: {}\n".format(scene.x))
        f.write("Actuator IDs: {}\n".format(scene.actuator_id))
        f.write("Particle types: {}\n".format(scene.particle_type))
        f.write("\n")

    initial_x = np.array(scene.x, dtype=np.float32)

    # Create the initial population.
    pop_size = options.pop
    n_generations = options.gens
    population = []
    for k in range(pop_size):
        candidate = {}
        candidate["weights"] = (np.random.randn(n_actuators, n_sin_waves).astype(np.float32) * 0.01)
        candidate["bias"] = (np.random.randn(n_actuators).astype(np.float32) * 0.01)
        population.append(candidate)

    # Record the initial population parameters.
    with open(params_file_path, 'a') as f:
        f.write("Initial Population Parameters:\n")
        for idx, candidate in enumerate(population):
            f.write("Candidate {}: weights: {} | bias: {}\n".format(
                idx,
                candidate["weights"].tolist(),
                candidate["bias"].tolist()))
        f.write("\n")

    def load_candidate(candidate):
        """Loads a candidate's parameters into the simulation fields."""
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] = candidate["weights"][i, j]
            bias[i] = candidate["bias"][i]

    def reset_simulation():
        """Resets the simulation to its initial state."""
        for i in range(scene.n_particles):
            x[0, i] = initial_x[i]
            F[0, i] = [[1.0, 0.0], [0.0, 1.0]]
            actuator_id[i] = scene.actuator_id[i]
            particle_type[i] = scene.particle_type[i]
        x_avg[None] = [0, 0]
        loss[None] = 0.0

    # Save the "before optimization" video using the best candidate from the initial population.
    import copy
    initial_candidate = copy.deepcopy(population[0])
    load_candidate(initial_candidate)
    reset_simulation()
    frames_before = forward(1500, record=True, record_interval=5)
    video_before_path = os.path.join(TRIAL_NUM, f"{TRIAL_NUM}_before.mp4")
    imageio.mimwrite(video_before_path, frames_before, fps=30)
    print("Saved video before optimization to", video_before_path)

    best_loss_over_gens = []
    mutation_std = 0.05
    num_elites = max(1, int(0.2 * pop_size))  # Preserve top 20%

    for gen in range(n_generations):
        candidate_losses = []
        # Evaluate each candidate in the population.
        for candidate in population:
            load_candidate(candidate)
            reset_simulation()
            forward()  # Run simulation (using default steps)
            candidate_loss = loss[None]
            candidate["loss"] = candidate_loss
            candidate_losses.append(candidate_loss)
        # Sort population by loss (lower is better, since loss = -distance traveled)
        population.sort(key=lambda c: c["loss"])
        best_loss = population[0]["loss"]
        best_loss_over_gens.append(best_loss)
        print("Generation", gen, "best loss:", best_loss)

        # Every 10 generations, record parameters and save a visualization.
        if gen % 10 == 0:
            with open(params_file_path, 'a') as f:
                f.write("Generation {}: best loss: {}\n".format(gen, best_loss))
                f.write("Best candidate weights: {}\n".format(population[0]["weights"].tolist()))
                f.write("Best candidate bias: {}\n".format(population[0]["bias"].tolist()))
                f.write("\n")
            load_candidate(population[0])
            reset_simulation()
            forward(1500)  # Run simulation to update state.
            save_visualization(gen)

        # Reproduction: keep elites and fill the population with mutated offspring.
        elites = population[:num_elites]
        new_population = [{"weights": np.copy(c["weights"]), "bias": np.copy(c["bias"])} for c in elites]
        while len(new_population) < pop_size:
            parent = random.choice(elites)
            child = {"weights": np.copy(parent["weights"]), "bias": np.copy(parent["bias"])}
            child["weights"] += np.random.randn(*child["weights"].shape).astype(np.float32) * mutation_std
            child["bias"] += np.random.randn(*child["bias"].shape).astype(np.float32) * mutation_std
            new_population.append(child)
        population = new_population

    # Save the "after optimization" video using the final candidate.
    load_candidate(population[0])
    reset_simulation()
    frames_after = forward(1500, record=True, record_interval=5)
    video_after_path = os.path.join(TRIAL_NUM, f"{TRIAL_NUM}_after.mp4")
    imageio.mimwrite(video_after_path, frames_after, fps=30)
    print("Saved video after optimization to", video_after_path)

    # Re-evaluate the final best candidate (as in the old code) to update its loss.
    load_candidate(population[0])
    reset_simulation()
    forward()  # Run simulation without recording to update loss
    population[0]["loss"] = loss[None]

    # Append final generation's details to the params file.
    with open(params_file_path, 'a') as f:
        f.write("Final Generation (Generation {}):\n".format(n_generations - 1))
        f.write("Best candidate loss: {}\n".format(population[0]["loss"]))
        f.write("Best candidate weights: {}\n".format(population[0]["weights"].tolist()))
        f.write("Best candidate bias: {}\n".format(population[0]["bias"].tolist()))
        f.write("\n")

    # Plot the loss curve.
    plt.figure()
    plt.title("Evolutionary Optimization Loss")
    plt.ylabel("Loss")
    plt.xlabel("Generations")
    plt.plot(best_loss_over_gens)
    loss_graph_path = os.path.join(TRIAL_NUM, f"{TRIAL_NUM}_loss.png")
    plt.savefig(loss_graph_path)
    plt.show()

if __name__ == '__main__':
    main()
