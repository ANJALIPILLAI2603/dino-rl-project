# environment.py
"""
DinoEnvironment
===============
All game physics, obstacle logic, collision detection, and state encoding.
Completely independent of Pygame — can run headless for fast training.
"""

import random


class DinoEnvironment:
    # ── World dimensions ──────────────────────────────────────────────────
    GAME_WIDTH  = 400      # Width of the scrolling world (pixels)
    GAME_HEIGHT = 300      # Visible height
    GROUND_Y    = 245      # Y-coordinate where the dino's feet rest
    DINO_X      = 70       # Dino's fixed horizontal position

    # ── Hitbox size ───────────────────────────────────────────────────────
    DINO_W = 22
    DINO_H = 38

    # ── Physics ───────────────────────────────────────────────────────────
    GRAVITY       =  0.75   # Downward acceleration (px / frame²)
    JUMP_VEL      = -13.0   # Initial upward velocity when jumping
    INITIAL_SPEED =  5.0    # Obstacle scroll speed at episode start
    MAX_SPEED     = 11.0    # Speed ceiling

    def __init__(self):
        self.reset()

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def reset(self):
        """
        Start a new episode.
        Returns the initial *discrete* state (int).
        """
        self.dino_y    = float(self.GROUND_Y)
        self.dino_vy   = 0.0
        self.on_ground = True

        self.obstacles      = []           # list of {"x", "w", "h"} dicts
        self.speed          = self.INITIAL_SPEED
        self.score          = 0
        self.done           = False

        self.spawn_timer    = 0
        self.spawn_interval = random.randint(80, 150)

        return self.get_discrete_state()

    # ── Step functions (one for each state type) ──────────────────────────

    def step(self, action):
        """
        Advance simulation one frame.  Returns a *discrete* state.

        Parameters
        ----------
        action : int — 0 = do nothing, 1 = jump

        Returns
        -------
        state  : int    — discrete state index  (used by Q-Learning / SARSA)
        reward : float  — +1 per frame survived, -100 on collision
        done   : bool
        """
        self._apply_action(action)
        self._physics_tick()
        self._update_obstacles()

        if self._collides():
            self.done = True
            return self.get_discrete_state(), -100.0, True

        self.score += 1
        self.speed  = min(self.INITIAL_SPEED + self.score / 500.0, self.MAX_SPEED)
        return self.get_discrete_state(), 1.0, False

    def step_continuous(self, action):
        """
        Same as step() but returns a *continuous* state vector.
        Used by the DQN agent.

        Returns
        -------
        state  : list[float] — 5-element feature vector
        reward : float
        done   : bool
        """
        self._apply_action(action)
        self._physics_tick()
        self._update_obstacles()

        if self._collides():
            self.done = True
            return self.get_continuous_state(), -100.0, True

        self.score += 1
        self.speed  = min(self.INITIAL_SPEED + self.score / 500.0, self.MAX_SPEED)
        return self.get_continuous_state(), 1.0, False

    # ── State encoders ────────────────────────────────────────────────────

    def get_discrete_state(self):
        """
        Encodes state as a single integer in [0, 21].

        Encoding:
          dist_bucket (0–10) × 2  +  is_jumping (0 or 1)

          • Buckets 0–9 : distance to nearest obstacle split into 10 bands
          • Bucket 10   : no obstacle on screen
          • is_jumping  : 1 if dino is airborne
        """
        nearest = self._nearest_obstacle()

        if nearest is None:
            dist_bucket = 10                    # nothing to worry about
        else:
            dist = max(0.0, nearest["x"] - self.DINO_X)
            dist_bucket = min(int(dist / (self.GAME_WIDTH / 10)), 9)

        is_jumping = 0 if self.on_ground else 1
        return dist_bucket * 2 + is_jumping     # range: 0 … 21

    def get_continuous_state(self):
        """
        Returns a 5-float feature vector for DQN.
        All values are roughly normalised to [0, 1].

        [dist_norm, obstacle_height_norm, dino_airtime_norm, vy_norm, speed_norm]
        """
        nearest = self._nearest_obstacle()

        if nearest is None:
            dist_norm  = 1.0
            obs_h_norm = 0.0
        else:
            dist_norm  = max(0.0, nearest["x"] - self.DINO_X) / self.GAME_WIDTH
            obs_h_norm = nearest["h"] / 65.0

        # How high the dino currently is (0 = on ground, 1 = max jump height)
        height_norm = max(0.0, self.GROUND_Y - self.dino_y) / self.GROUND_Y

        # Vertical velocity normalised
        vy_norm = max(0.0, min(1.0, (self.dino_vy + 15.0) / 30.0))

        speed_range = max(self.MAX_SPEED - self.INITIAL_SPEED, 1e-6)
        speed_norm  = (self.speed - self.INITIAL_SPEED) / speed_range

        return [dist_norm, obs_h_norm, height_norm, vy_norm, speed_norm]

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _apply_action(self, action):
        """Trigger a jump if requested and dino is on the ground."""
        if action == 1 and self.on_ground:
            self.dino_vy   = self.JUMP_VEL
            self.on_ground = False

    def _physics_tick(self):
        """Apply gravity and ground clamping."""
        self.dino_vy += self.GRAVITY
        self.dino_y  += self.dino_vy

        if self.dino_y >= self.GROUND_Y:
            self.dino_y    = float(self.GROUND_Y)
            self.dino_vy   = 0.0
            self.on_ground = True

    def _update_obstacles(self):
        """Spawn, move, and remove off-screen obstacles."""
        # Spawn a new cactus when the timer fires
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            h = random.choice([35, 48, 60])
            w = random.randint(20, 28)
            self.obstacles.append({"x": float(self.GAME_WIDTH + 10), "w": w, "h": h})
            self.spawn_timer    = 0
            self.spawn_interval = random.randint(70, 140)

        # Scroll all obstacles to the left
        for obs in self.obstacles:
            obs["x"] -= self.speed

        # Remove obstacles that have scrolled past the dino
        self.obstacles = [o for o in self.obstacles if o["x"] > -60]

    def _nearest_obstacle(self):
        """Return the closest obstacle that is still in front of the dino."""
        nearest = None
        for obs in self.obstacles:
            # Only consider obstacles the dino hasn't fully passed
            if obs["x"] + obs["w"] > self.DINO_X - 5:
                if nearest is None or obs["x"] < nearest["x"]:
                    nearest = obs
        return nearest

    def _collides(self):
        """
        Axis-aligned bounding-box collision with a small leniency margin
        (makes the game slightly forgiving so agents can focus on learning).
        """
        m  = 5                               # forgiveness margin (pixels)
        dx = self.DINO_X - self.DINO_W // 2 + m
        dy = self.dino_y - self.DINO_H      + m
        dw = self.DINO_W - 2 * m
        dh = self.DINO_H - 2 * m

        for obs in self.obstacles:
            ox = obs["x"]
            oy = self.GROUND_Y - obs["h"]
            if (dx < ox + obs["w"] and dx + dw > ox and
                    dy < oy + obs["h"] and dy + dh > oy):
                return True
        return False
