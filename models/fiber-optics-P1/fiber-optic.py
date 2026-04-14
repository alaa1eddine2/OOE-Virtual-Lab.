# fiber_bend_simulation.py
# Interactive simulation of bend-induced loss in optical fiber
# Based on empirical model: L_R = 70 * exp(-0.5 * R) [dB/turn] at λ=1550 nm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# ----------------------------
# Constants from the paper
# ----------------------------
ETA_R1 = 70.0  # dB/turn coefficient
ETA_R2 = 0.5  # 1/mm

# Material presets (n1, n2)
MATERIAL_PRESETS = {
    "Standard SMF": (1.48, 1.46),
    "High NA Fiber": (1.50, 1.46),
    "Plastic Fiber": (1.49, 1.40),
}

# Visual scaling (not physical)
VIS_CORE_RADIUS = 6.0
VIS_CLAD_RADIUS = 18.0


# ----------------------------
# Helper functions
# ----------------------------
def loss_to_escape_probability(loss_db):
    """Convert dB loss to fraction of power lost."""
    if loss_db <= 0:
        return 0.0
    transmitted = 10 ** (-loss_db / 10.0)
    return max(0.0, min(1.0, 1.0 - transmitted))


def calculate_data_transfer(loss_db):
    """Convert loss to data transfer percentage (0-100%)."""
    escape_prob = loss_to_escape_probability(loss_db)
    return 100.0 * (1.0 - escape_prob)


def calculate_loss(radius_mm):
    """Calculate loss using paper's equation (12)."""
    return ETA_R1 * np.exp(-ETA_R2 * radius_mm)


# ----------------------------
# Main App Class
# ----------------------------
class FiberBendSimulator:
    def __init__(self):
        self.radius_mm = 8.0
        self.speed_factor = 1.0
        self.material_key = "Standard SMF"
        self.is_paused = False
        self.time_phase = 0.0
        self.pulses = []  # list of (creation_time, escaped: bool, x, y, segment)
        self.historical_data = []  # Store all (radius, data_transfer) points
        self.bend_center = (0, 0)  # Will be updated based on radius

        # Setup figure with 2 panels
        self.fig = plt.figure(figsize=(12, 6.5))
        self.fig.suptitle("Fiber-Optic Bend Simulation", fontsize=16, fontweight='bold')

        # Left panel: Data transfer graph (with historical data)
        self.ax_graph = plt.axes([0.05, 0.35, 0.35, 0.55])
        self.ax_graph.set_title("Data Transfer vs. Bend Radius", fontsize=12)
        self.ax_graph.set_xlim(4, 15)
        self.ax_graph.set_ylim(0, 100)
        self.ax_graph.set_xlabel("Bend Radius (mm)")
        self.ax_graph.set_ylabel("Data Transfer (%)")
        self.ax_graph.grid(True, linestyle='--', alpha=0.3)
        self.ax_graph.plot([4, 15], [100, 100], 'k--', alpha=0.3)
        self.ax_graph.plot([4, 15], [0, 0], 'k--', alpha=0.3)

        # Historical data plot
        self.historical_line, = self.ax_graph.plot([], [], 'b-', linewidth=2)
        self.data_points, = self.ax_graph.plot([], [], 'bo', markersize=8)

        # Right panel: L-shaped cable with animation
        self.ax_animation = plt.axes([0.45, 0.35, 0.5, 0.55])
        self.ax_animation.set_title("Light Propagation & Loss", fontsize=12)
        self.ax_animation.set_xlim(-150, 150)
        self.ax_animation.set_ylim(-50, 150)
        self.ax_animation.set_aspect('equal')
        self.ax_animation.axis('off')

        # Top controls
        self.ax_radius = plt.axes([0.2, 0.25, 0.3, 0.03])
        self.ax_speed = plt.axes([0.2, 0.20, 0.3, 0.03])
        self.ax_material = plt.axes([0.6, 0.20, 0.25, 0.12])
        self.ax_start_stop = plt.axes([0.88, 0.25, 0.1, 0.04])

        # Create controls
        self.slider_radius = Slider(self.ax_radius, 'Bend Radius (mm)', 4.0, 15.0, valinit=self.radius_mm, valstep=0.1)
        self.slider_speed = Slider(self.ax_speed, 'Speed', 0.2, 3.0, valinit=self.speed_factor, valstep=0.1)
        self.radio_material = RadioButtons(self.ax_material, list(MATERIAL_PRESETS.keys()), active=0)
        self.button_toggle = Button(self.ax_start_stop, 'Start')

        # Connect events
        self.slider_radius.on_changed(self.update_radius)
        self.slider_speed.on_changed(self.update_speed)
        self.radio_material.on_clicked(self.update_material)
        self.button_toggle.on_clicked(self.toggle_animation)

        # Initial draw
        self.draw_cable()
        self.update_data_graph()

        # Animation setup
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

    def update_radius(self, val):
        self.radius_mm = float(val)
        self.update_data_graph()
        self.draw_cable()

    def update_speed(self, val):
        self.speed_factor = float(val)

    def update_material(self, label):
        self.material_key = label

    def toggle_animation(self, _):
        self.is_paused = not self.is_paused
        self.button_toggle.label.set_text('Pause' if not self.is_paused else 'Start')
        plt.draw()

    def update_data_graph(self):
        """Update the data transfer graph with current values."""
        loss_db = calculate_loss(self.radius_mm)
        data_transfer = calculate_data_transfer(loss_db)

        # Add to historical data (avoid duplicates within 0.2mm tolerance)
        exists = any(abs(r - self.radius_mm) < 0.2 and abs(t - data_transfer) < 0.5
                     for r, t in self.historical_data)
        if not exists:
            self.historical_data.append((self.radius_mm, data_transfer))
            self.historical_data.sort(key=lambda x: x[0])

        # Update the data points
        radii, transfers = zip(*self.historical_data) if self.historical_data else ([], [])
        self.data_points.set_data(radii, transfers)
        self.historical_line.set_data(radii, transfers)

        # Update title with current values
        self.ax_graph.set_title(
            f"Data Transfer vs. Bend Radius | R = {self.radius_mm:.1f} mm | Transfer: {data_transfer:.1f}%",
            fontsize=12
        )

    def draw_cable(self):
        """Draw L-shaped cable with rounded corner (bend center at 135°)"""
        self.ax_animation.clear()
        self.ax_animation.set_title("Light Propagation & Loss", fontsize=12)
        self.ax_animation.set_xlim(-150, 150)
        self.ax_animation.set_ylim(-50, 150)
        self.ax_animation.set_aspect('equal')
        self.ax_animation.axis('off')

        # Calculate bend center at 135° (like your reference image)
        bend_radius = self.radius_mm * 5.0  # Visual scaling factor
        self.bend_center = (-bend_radius, 50 + bend_radius)  # Center at 135° position

        # Draw straight horizontal segment (left part)
        self.ax_animation.plot([-120, 0], [50, 50], 'k-', linewidth=2.5, zorder=1)

        # Draw bend circle (center at bend_center)
        circle = Circle(
            self.bend_center,
            bend_radius,
            fill=False,
            linestyle='--',
            color='gray',
            alpha=0.4,
            zorder=0
        )
        self.ax_animation.add_patch(circle)

        # Draw bend radius label (from center to outer edge)
        self.ax_animation.plot([self.bend_center[0], 0],
                               [self.bend_center[1], 50],
                               'k-', linewidth=1, alpha=0.7)
        self.ax_animation.plot([-2, 2],
                               [50, 50],
                               'k-', linewidth=1)
        self.ax_animation.text(
            0, 50 + 8,
            f"R = {self.radius_mm:.1f} mm",
            fontsize=10,
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

        # Draw center point
        self.ax_animation.plot(self.bend_center[0], self.bend_center[1], 'ko', markersize=5, zorder=2)
        self.ax_animation.text(
            self.bend_center[0] + 8, self.bend_center[1] - 5,
            "Bend Center",
            fontsize=9,
            style='italic'
        )

        # Draw curved segment (bend) - quarter circle from left to up
        theta = np.linspace(-np.pi/2, 0, 100)
        x = self.bend_center[0] + bend_radius * np.cos(theta)
        y = self.bend_center[1] + bend_radius * np.sin(theta)
        self.ax_animation.plot(x, y, 'k-', linewidth=2.5, zorder=1)

        # Draw straight vertical segment (top part)
        self.ax_animation.plot([0, 0], [50, 50 + bend_radius], 'k-', linewidth=2.5, zorder=1)

        # Draw core/cladding (thickened for visibility)
        # Horizontal segment - core (cyan) inside cladding (blue)
        self.ax_animation.plot([-120, 0], [50, 50], 'b-', linewidth=8, alpha=0.6, zorder=1)
        self.ax_animation.plot([-120, 0], [50, 50], 'c-', linewidth=5, alpha=0.9, zorder=2)

        # Curved segment
        theta = np.linspace(-np.pi/2, 0, 100)
        x_core = self.bend_center[0] + (bend_radius - VIS_CORE_RADIUS) * np.cos(theta)
        y_core = self.bend_center[1] + (bend_radius - VIS_CORE_RADIUS) * np.sin(theta)
        x_clad = self.bend_center[0] + (bend_radius - VIS_CLAD_RADIUS) * np.cos(theta)
        y_clad = self.bend_center[1] + (bend_radius - VIS_CLAD_RADIUS) * np.sin(theta)
        self.ax_animation.plot(x_clad, y_clad, 'b-', linewidth=8, alpha=0.6, zorder=1)
        self.ax_animation.plot(x_core, y_core, 'c-', linewidth=5, alpha=0.9, zorder=2)

        # Vertical segment
        self.ax_animation.plot([0, 0], [50, 50 + bend_radius], 'b-', linewidth=8, alpha=0.6, zorder=1)
        self.ax_animation.plot([0, 0], [50, 50 + bend_radius], 'c-', linewidth=5, alpha=0.9, zorder=2)

    def animate(self, _):
        if self.is_paused:
            return

        # Clear previous animation elements (keep cable structure)
        for collection in self.ax_animation.collections[:]:
            collection.remove()

        # Compute loss and escape probability
        loss_db = calculate_loss(self.radius_mm)
        escape_prob = loss_to_escape_probability(loss_db)
        bend_radius = self.radius_mm * 5.0

        # Add new pulse periodically
        if not self.pulses or (self.time_phase - self.pulses[-1][0]) > 0.45:
            if len(self.pulses) < 25:
                escaped = np.random.rand() < escape_prob
                self.pulses.append((self.time_phase, escaped, -120, 50, 'horizontal'))

        # Update and draw each pulse
        new_pulses = []
        for t0, escaped, x, y, segment in self.pulses:
            # Move pulse along cable path
            if segment == 'horizontal':
                # Move along horizontal segment
                x += 2.5 * self.speed_factor
                if x >= 0:
                    segment = 'bend'
                    x = 0
                    y = 50

            # Bend segment (quarter circle)
            if segment == 'bend':
                progress = (self.time_phase - t0) * 0.1 * self.speed_factor
                angle = -np.pi / 2 + progress

                if angle >= 0:
                    # Switch cleanly to vertical and STOP bend drawing
                    segment = 'vertical'
                    x = 0
                    y = 50
                    new_pulses.append((t0, escaped, x, y, segment))
                    continue
                else:
                    x = self.bend_center[0] + bend_radius * np.cos(angle)
                    y = self.bend_center[1] + bend_radius * np.sin(angle)

            # Vertical segment
            if segment == 'vertical':
                y += 2.5 * self.speed_factor

            # Remove if exited
            if y > 130 or x < -130:
                continue

            # Store updated pulse
            new_pulses.append((t0, escaped, x, y, segment))

            # Draw pulse
            if escaped:
                # Drift outward from cable (not from bend circle)
                if segment == 'bend':
                    # Calculate outward direction from cable
                    dx = x - self.bend_center[0]
                    dy = y - self.bend_center[1]
                    norm = np.sqrt(dx * dx + dy * dy) + 1e-5
                    drift_x = x + (dx / norm) * np.random.uniform(8, 15)
                    drift_y = y + (dy / norm) * np.random.uniform(8, 15)
                else:
                    # Horizontal segment: drift up
                    if segment == 'horizontal':
                        drift_x = x + np.random.uniform(-3, 8)
                        drift_y = y + np.random.uniform(5, 12)
                    # Vertical segment: drift right
                    else:
                        drift_x = x + np.random.uniform(5, 12)
                        drift_y = y + np.random.uniform(-3, 8)

                self.ax_animation.scatter(
                    [drift_x], [drift_y],
                    c='#FF0000',
                    s=40,
                    edgecolor='#8B0000',
                    zorder=5,
                    alpha=0.85,
                    marker='o'
                )
            else:
                # Guided pulse inside core (on cable path)
                self.ax_animation.scatter(
                    [x], [y],
                    c='#00FFFF',
                    s=30,
                    edgecolor='white',
                    zorder=4,
                    alpha=0.95
                )

        self.pulses = new_pulses
        self.time_phase += 0.05 * self.speed_factor


# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    sim = FiberBendSimulator()
    plt.show()