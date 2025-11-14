# This is SCRIPT 2: The NEW Fluid Visualizer
#
# This script reads the 'hybrid_analysis_data.npz' file.
# It uses a particle system to create a "fluid" simulation.
#
# - AI Data controls particle COLOR.
# - Beat Data triggers a "splash" and "shake".
# - Frequency Data controls *where* particles spawn (bass=left, treble=right).
# - Volume Data controls *how many* particles spawn (loud=more).
#
# NEW:
# - Bass-heavy sound = "Lava" (slow, gooey, red/orange)
# - Treble-heavy sound = "Water" (fast, splashy, blue/cyan)
# - AI color is blended on top of the element color.

import pygame
import numpy as np
import sys
import random

# --- Configuration ---
INPUT_DATA_FILE = "hybrid_analysis_data.npz"

# Graphics settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (10, 10, 30)
SHOCKWAVE_COLOR = (255, 255, 255)
MAX_PARTICLES = 5000 

# --- NEW: Element & Physics Config ---
SHAKE_INTENSITY_ON_BEAT = 20
SHAKE_DECAY = 0.80

# "Water" (treble) physics
WATER_GRAVITY = 0.5
WATER_DRAG = 0.96
WATER_KICK = -15
WATER_LIFESPAN = 100
WATER_COLOR_BASE = (0, 60, 255)

# "Lava" (bass) physics
LAVA_GRAVITY = 0.2
LAVA_DRAG = 0.99
LAVA_KICK = -8
LAVA_LIFESPAN = 240
LAVA_COLOR_BASE = (255, 60, 0)

# --- Helper Function: Linear Interpolation (lerp) ---
def lerp(a, b, t):
    """Linearly interpolate from a to b by t (t=0.0 -> a, t=1.0 -> b)"""
    return a + (b - a) * t

def lerp_color(color_a, color_b, t):
    """Lerp between two (R, G, B) colors"""
    r = int(lerp(color_a[0], color_b[0], t))
    g = int(lerp(color_a[1], color_b[1], t))
    b = int(lerp(color_a[2], color_b[2], t))
    return (r, g, b)

# --- Particle Class (Updated) ---
class Particle:
    def __init__(self, x, y, vx, vy, color, size, lifespan):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self, current_gravity, current_drag):
        # Apply physics passed in from the main loop
        self.x += self.vx
        self.y += self.vy
        self.vy += current_gravity
        
        self.vx *= current_drag
        
        self.lifespan -= 1
        
    def draw(self, surface):
        alpha = int(255 * (self.lifespan / self.initial_lifespan))
        alpha = max(0, min(255, alpha))
        
        temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        
        # Use a slightly brighter version of the color for the core
        core_color = (min(255, self.color[0] + 50), 
                      min(255, self.color[1] + 50), 
                      min(255, self.color[2] + 50))
        
        # Draw a bright core with low alpha
        pygame.draw.circle(
            temp_surf, 
            (*core_color, int(alpha * 0.1)), # Very faint "goo"
            (self.size, self.size), 
            self.size
        )
        # Draw a smaller, sharper core
        pygame.draw.circle(
            temp_surf, 
            (*core_color, int(alpha * 0.5)), 
            (self.size, self.size), 
            int(self.size * 0.5)
        )
        
        blit_pos = (int(self.x - self.size), int(self.y - self.size))
        # Use BLEND_RGBA_ADD for a "glowing" effect
        surface.blit(temp_surf, blit_pos, special_flags=pygame.BLEND_RGBA_ADD)

# --- Main Visualizer ---
def run_visualizer():
    # --- 1. Load Analysis Data ---
    print(f"Loading hybrid analysis data from {INPUT_DATA_FILE}...")
    try:
        data = np.load(INPUT_DATA_FILE, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: File '{INPUT_DATA_FILE}' not found.")
        print("Please run 'hybrid_analyzer.py' first.")
        sys.exit()

    mel_spec = data['mel_spec']
    rms = data['rms']
    beat_frames = data['beat_frames']
    VIS_FPS = float(data['vis_fps'])
    AUDIO_FILE = str(data['audio_file'])
    BPM = float(data['tempo'])
    rgb_data = data['rgb_data']
    
    NUM_BINS = mel_spec.shape[0]
    TOTAL_FRAMES = mel_spec.shape[1]
    max_rms = np.max(rms) or 1.0
    beat_set = set(beat_frames)
    
    print("Data loaded. Starting fluid visualizer...")

    # --- 2. Initialize Pygame and Audio ---
    pygame.init()
    pygame.font.init()
    pygame.mixer.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    render_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    particle_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    
    pygame.display.set_caption("Fluid Visualizer (Lava/Water + AI Color)")
    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.SysFont('Arial', 30, bold=True)
    except:
        font = pygame.font.Font(pygame.font.get_default_font(), 30)

    try:
        pygame.mixer.music.load(AUDIO_FILE)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing audio file: {e}")
        pygame.quit()
        sys.exit()

    # --- 3. Main Loop ---
    running = True
    current_frame = 0
    particles = []
    
    # Physics Variables
    current_shake_intensity = 0
    shockwave_radius = 0
    shockwave_alpha = 0
    
    # NEW: Element ratio. 0.0 = Water, 1.0 = Lava
    current_element_ratio = 0.5 # Start in the middle
    
    while running and current_frame < TOTAL_FRAMES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        if not pygame.mixer.music.get_busy() and current_frame > 100:
            running = False

        # --- 1. GET DATA FOR THIS FRAME ---
        current_rms_normalized = (rms[0, current_frame] / max_rms)
        current_mel_bin = (mel_spec[:, current_frame] + 80) / 80 # Normalized 0-1
        current_ai_color = rgb_data[current_frame] # Get AI color
        
        is_beat = current_frame in beat_set
        
        # --- 2. CALCULATE ELEMENT & PHYSICS ---
        
        # Calculate bass vs treble energy
        bass_energy = np.mean(current_mel_bin[0:25])  # First ~20% of bins
        treble_energy = np.mean(current_mel_bin[80:128]) # Last ~40% of bins
        total_energy = bass_energy + treble_energy + 1e-6 # Add epsilon to avoid divide by zero
        
        # 0.0 = all treble (Water), 1.0 = all bass (Lava)
        target_element_ratio = bass_energy / total_energy
        
        # Smoothly move towards the target ratio (95% old, 5% new)
        current_element_ratio = (current_element_ratio * 0.95) + (target_element_ratio * 0.05)

        # Interpolate all our physics based on the element ratio
        current_gravity = lerp(WATER_GRAVITY, LAVA_GRAVITY, current_element_ratio)
        current_drag = lerp(WATER_DRAG, LAVA_DRAG, current_element_ratio)
        current_lifespan = lerp(WATER_LIFESPAN, LAVA_LIFESPAN, current_element_ratio)
        current_kick = lerp(WATER_KICK, LAVA_KICK, current_element_ratio)
        current_base_color = lerp_color(WATER_COLOR_BASE, LAVA_COLOR_BASE, current_element_ratio)

        # --- 3. FADE & APPLY BEAT REACTIONS ---
        current_shake_intensity = max(0, current_shake_intensity * SHAKE_DECAY)
        shockwave_radius += 20
        shockwave_alpha = max(0, shockwave_alpha - 15)
        
        if is_beat:
            current_shake_intensity = SHAKE_INTENSITY_ON_BEAT
            shockwave_radius = 50
            shockwave_alpha = 255
            
            # "Splash" - kick all existing particles
            for p in particles:
                p.vy += current_kick # Use dynamic kick
                p.vx += random.uniform(-2, 2)
        
        # --- 4. SPAWN NEW PARTICLES ---
        if len(particles) < MAX_PARTICLES:
            num_to_spawn = int(current_rms_normalized * 15)
            
            for _ in range(num_to_spawn):
                try:
                    hot_bin = random.choices(range(NUM_BINS), weights=current_mel_bin, k=1)[0]
                except:
                    hot_bin = random.randint(0, NUM_BINS - 1)
                
                spawn_x = (hot_bin / NUM_BINS) * SCREEN_WIDTH
                energy = np.clip(current_mel_bin[hot_bin], 0.1, 1.0)
                
                # --- NEW COLOR LOGIC ---
                # Blend the AI color (mood) with the Element color (base)
                # 30% Element, 70% AI (AI color is now dominant)
                final_color = lerp_color(current_base_color, current_ai_color, 0.7)

                p = Particle(
                    x = spawn_x + random.uniform(-10, 10),
                    y = SCREEN_HEIGHT - 20,
                    vx = random.uniform(-2, 2),
                    vy = random.uniform(-5, -12) * energy,
                    color = final_color,
                    size = int(energy * 10) + 4,
                    lifespan = current_lifespan # Use dynamic lifespan
                )
                particles.append(p)

        # --- 5. DRAW TO OFF-SCREEN SURFACES ---
        render_surface.fill(BG_COLOR)
        particle_surface.fill((0, 0, 0, 0))

        # Update and draw particles
        for i in range(len(particles) - 1, -1, -1):
            p = particles[i]
            # Pass in current physics
            p.update(current_gravity, current_drag) 
            
            if p.lifespan <= 0 or p.y > SCREEN_HEIGHT + p.size:
                particles.pop(i)
            else:
                p.draw(particle_surface)
        
        render_surface.blit(particle_surface, (0, 0))

        # --- Draw Shockwave ---
        if shockwave_alpha > 0:
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(
                s, 
                (*SHOCKWAVE_COLOR, int(shockwave_alpha)),
                (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), 
                int(shockwave_radius), 
                5
            )
            render_surface.blit(s, (0, 0))

        # --- Draw BPM Text ---
        bpm_text = font.render(f'BPM: {BPM:.0f}', True, (255, 255, 255))
        render_surface.blit(bpm_text, (20, 20))
        
        # --- NEW: Draw Element Status ---
        element_text = "LAVA" if current_element_ratio > 0.5 else "WATER"
        element_color = lerp_color(WATER_COLOR_BASE, LAVA_COLOR_BASE, current_element_ratio)
        status_text = font.render(f'MODE: {element_text}', True, element_color)
        render_surface.blit(status_text, (20, 60))

        # --- 6. APPLY SHAKE AND DRAW TO SCREEN ---
        screen.fill(BG_COLOR)
        if current_shake_intensity > 1:
            shake_x = random.randint(-int(current_shake_intensity), int(current_shake_intensity))
            shake_y = random.randint(-int(current_shake_intensity), int(current_shake_intensity))
        else:
            shake_x, shake_y = 0, 0
        
        screen.blit(render_surface, (shake_x, shake_y))
        pygame.display.flip()
        
        clock.tick(VIS_FPS)
        current_frame += 1

    # --- 7. Cleanup ---
    print("Shutting down...")
    pygame.mixer.music.stop()
    pygame.quit()

if __name__ == "__main__":
    run_visualizer()