import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Physical Constants ---
m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
g = 9.81
dt = 0.02  # Time step for real-time integration

# --- State Variables ---
# [theta1, d_theta1, theta2, d_theta2]
state = np.array([np.pi/4, 0.0, np.pi/2, 0.0])
cart_pos = np.array([0.0, 0.0])
cart_vel = np.array([0.0, 0.0])

def get_derivatives(theta1, w1, theta2, w2, ax, ay):
    """
    Calculates angular accelerations based on cart acceleration (ax, ay).
    """
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s12, c12 = np.sin(theta1 - theta2), np.cos(theta1 - theta2)

    # Mass Matrix for the two pendulum arms
    M = np.array([
        [(m1 + m2) * L1**2, m2 * L1 * L2 * c12],
        [m2 * L1 * L2 * c12, m2 * L2**2]
    ])

    # RHS includes gravity, centripetal forces, and Fictitious Forces from Cart Accel
    # Note: Cart acceleration (ax, ay) acts as an inertial force on the pivots
    rhs = np.array([
        -m2 * L1 * L2 * w2**2 * s12 + (m1 + m2) * g * L1 * s1 - (m1 + m2) * L1 * (ax * c1 + ay * s1),
        m2 * L1 * L2 * w1**2 * s12 + m2 * g * L2 * s2 - m2 * L2 * (ax * c2 + ay * s2)
    ])

    accels = np.linalg.solve(M, rhs)
    return accels[0], accels[1]

# --- UI / Interaction Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_title("Click and Drag the Red Cart to Sling the Pendulum")

line, = ax.plot([], [], 'o-', lw=3, color='#2c3e50', markersize=10)
cart_p = plt.Circle((0, 0), 0.2, fc='red', zorder=5)
ax.add_patch(cart_p)

mouse_pos = np.array([0.0, 0.0])
is_dragging = [False]

def on_mouse_move(event):
    if event.inaxes:
        mouse_pos[0], mouse_pos[1] = event.xdata, event.ydata

def on_click(event): is_dragging[0] = True
def on_release(event): is_dragging[0] = False

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)

def update(frame):
    global state, cart_pos, cart_vel
    
    # 1. Update Cart (Mover-Gripper)
    if is_dragging[0]:
        new_pos = mouse_pos.copy()
        # Calculate approximate acceleration of the cart
        new_vel = (new_pos - cart_pos) / dt
        accel = (new_vel - cart_vel) / dt
        cart_pos, cart_vel = new_pos, new_vel
    else:
        accel = np.array([0.0, 0.0])
        cart_vel *= 0.9 # Friction/Damping for the cart when let go
        cart_pos += cart_vel * dt

    # 2. Integrate Pendulum Physics (Semi-Implicit Euler)
    th1, w1, th2, w2 = state
    dw1, dw2 = get_derivatives(th1, w1, th2, w2, accel[0], accel[1])
    
    # Damping for the joints (prevents infinite spinning)
    w1 = (w1 + dw1 * dt) * 0.995
    w2 = (w2 + dw2 * dt) * 0.995
    th1 += w1 * dt
    th2 += w2 * dt
    
    state = [th1, w1, th2, w2]

    # 3. Update Visuals
    x1, y1 = cart_pos[0] + L1*np.sin(th1), cart_pos[1] - L1*np.cos(th1)
    x2, y2 = x1 + L2*np.sin(th2), y1 - L2*np.cos(th2)
    
    cart_p.center = cart_pos
    line.set_data([cart_pos[0], x1, x2], [cart_pos[1], y1, y2])
    return line, cart_p

ani = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
plt.show()
