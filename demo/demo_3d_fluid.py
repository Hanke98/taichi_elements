import taichi as ti
import numpy as np
import os
import utils
from engine.mpm_solver import MPMSolver

write_to_disk = True

if write_to_disk:
    os.makedirs('outputs', exist_ok=True)

# Try to run on GPU
ti.init(arch=ti.cuda, device_memory_fraction=0.9)

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

R = 256
max_num_particles = 400**3

mpm = MPMSolver(res=(R, R, R), use_g2p2g=True, max_num_particles=max_num_particles, size=1)

mpm.set_gravity((0, -20, 0))

# mpm.add_sphere_collider(center=(0.25, 0.25, 0.5),
#                         radius=0.1,
#                         surface=mpm.surface_slip)
# mpm.add_sphere_collider(center=(0.5, 0.25, 0.5),
#                         radius=0.1,
#                         surface=mpm.surface_sticky)
# mpm.add_sphere_collider(center=(0.75, 0.25, 0.5),
#                         radius=0.1,
#                         surface=mpm.surface_separate)

# mpm.add_cube((0.3, 0.25, 0.5), (0.4, 0.2, 0.2),
#                 mpm.material_water,
#                 sample_density=1.0,
#                 color=0x8888FF)

radius = 0.05
for frame in range(512):

    mpm.add_ellipsoid((0.2, 0.25, 0.5), radius, mpm.material_water, color=0x8888FF, velocity=(2.0, 0.0, 0.0))
    mpm.add_ellipsoid((0.8, 0.25, 0.5), radius, mpm.material_water, color=0x8888FF, velocity=(-2.0, 0.0, 0.0))
    mpm.step(4e-3)
    # mpm.init_ggui_window()
    # mpm.draw_ggui(frame, "outputs/")
    particles = mpm.particle_info()
    np_x = particles['position'] / 1.0
    print(f'num particles: {len(np_x)}')

    # simple camera transform
    # screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    # screen_y = (np_x[:, 1])

    screen_x = np_x[:, 0]
    screen_y = np_x[:, 1]

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=1.1, color=particles['color'])
    gui.show(f'outputs/{frame:06d}.png' if write_to_disk else None)
