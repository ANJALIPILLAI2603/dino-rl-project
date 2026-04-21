# game.py
"""
Pygame Visualiser
=================
Renders three agent panels side-by-side, each showing:
  • Scrolling sky + ground
  • Pixel-art dino in the agent's colour
  • Green cactus obstacles
  • Header : agent name  |  episode  |  current score  |  best score
  • Footer : scroll speed  |  epsilon  |  buffer size (DQN)
"""

import pygame
import sys

# ─── Colour palette ───────────────────────────────────────────────────────────
SKY         = (210, 235, 255)
GROUND      = ( 90,  70,  50)
GRASS       = ( 70, 155,  45)
CACTUS_DARK = ( 20, 110,  20)
CACTUS_MID  = ( 34, 139,  34)
CACTUS_LITE = ( 60, 170,  60)
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
DARK_BG     = ( 22,  22,  32)
DIVIDER_COL = ( 55,  55,  75)
CLOUD_COL   = (240, 245, 255)

# One distinct body colour per agent
AGENT_COLORS = {
    "Q-Learning": ( 65, 135, 210),   # Steel blue
    "SARSA"     : (215,  95,  45),   # Burnt orange
    "DQN"       : ( 70, 190, 105),   # Emerald green
}

# ─── Layout constants ─────────────────────────────────────────────────────────
PANEL_W   = 400     # Matches DinoEnvironment.GAME_WIDTH exactly → no x-scaling
PANEL_H   = 430
HEADER_H  =  55     # Agent name + stats bar
GAME_H    = 300     # Matches DinoEnvironment.GAME_HEIGHT
FOOTER_H  = PANEL_H - HEADER_H - GAME_H   # 75 px

SCREEN_W  = PANEL_W * 3
SCREEN_H  = PANEL_H

# Absolute y positions of key lines on screen
GAME_TOP   = HEADER_H                   # 55
GAME_BOT   = HEADER_H + GAME_H          # 355
FOOTER_TOP = GAME_BOT                   # 355

# Ground in game-world coordinates (must match GROUND_Y in environment.py)
GROUND_REL  = 245
GROUND_ABS  = GAME_TOP + GROUND_REL    # 300  ← dino feet screen-y when on ground

# Static cloud positions (relative to panel left edge, game-area top)
_CLOUDS = [(55, 22), (180, 12), (310, 28)]


# ─── Public API ───────────────────────────────────────────────────────────────

def init_display(title="Dino Runner — RL Agent Comparison"):
    """Initialise Pygame and return (screen, clock)."""
    pygame.init()
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()
    return screen, clock


def handle_events():
    """Return False if the user closes the window or presses Esc."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def draw_frame(screen, agents_data, episode_counts, best_scores, epsilons=None):
    """
    Render one complete frame.

    Parameters
    ----------
    agents_data    : list of {"name": str, "env": DinoEnvironment}
    episode_counts : list[int] — current episode per agent
    best_scores    : list[int] — best score per agent
    epsilons       : list[float] | None — exploration rate per agent
    """
    if epsilons is None:
        epsilons = [None, None, None]

    screen.fill(DARK_BG)

    for idx, data in enumerate(agents_data):
        px    = idx * PANEL_W
        env   = data["env"]
        name  = data["name"]
        color = AGENT_COLORS[name]

        _draw_header(screen, px, name, color,
                     episode_counts[idx], env.score, best_scores[idx])
        _draw_game_area(screen, px, env, color)
        _draw_footer(screen, px, color, env, epsilons[idx])

        # Divider between panels
        if idx > 0:
            pygame.draw.line(screen, DIVIDER_COL, (px, 0), (px, SCREEN_H), 3)

    pygame.display.flip()


# ─── Private drawing helpers ──────────────────────────────────────────────────

def _draw_header(screen, px, name, color, episode, score, best):
    """Coloured header bar with agent name, episode number, score and best."""
    pygame.draw.rect(screen, color, (px, 0, PANEL_W, HEADER_H))

    # Subtle inner shadow
    shadow = tuple(max(0, c - 30) for c in color)
    pygame.draw.rect(screen, shadow, (px, HEADER_H - 4, PANEL_W, 4))

    font_big   = _font(18, bold=True)
    font_small = _font(13)

    # Agent name — centred
    label = font_big.render(name, True, WHITE)
    screen.blit(label, label.get_rect(centerx=px + PANEL_W // 2, y=7))

    # Episode (left) and scores (right)
    ep_surf = font_small.render(f"Ep {episode:>4}", True, WHITE)
    sc_surf = font_small.render(f"Score {score:>5}  Best {best:>5}", True, WHITE)
    screen.blit(ep_surf, (px + 8,  HEADER_H - 20))
    screen.blit(sc_surf, (px + PANEL_W - sc_surf.get_width() - 8, HEADER_H - 20))


def _draw_game_area(screen, px, env, color):
    """Sky, clouds, ground strip, obstacles, and dino."""
    # ── Sky ───────────────────────────────────────────────────────────────
    pygame.draw.rect(screen, SKY, (px, GAME_TOP, PANEL_W, GAME_H))

    # ── Clouds (decorative, static) ───────────────────────────────────────
    for cx, cy in _CLOUDS:
        sx, sy = px + cx, GAME_TOP + cy
        pygame.draw.ellipse(screen, CLOUD_COL, (sx,      sy,      65, 22))
        pygame.draw.ellipse(screen, CLOUD_COL, (sx + 18, sy - 10, 40, 22))

    # ── Obstacles ─────────────────────────────────────────────────────────
    for obs in env.obstacles:
        ox = px + int(obs["x"])
        oy = GROUND_ABS - obs["h"]
        _draw_cactus(screen, ox, oy, obs["w"], obs["h"])

    # ── Ground ────────────────────────────────────────────────────────────
    pygame.draw.rect(screen, GROUND, (px, GROUND_ABS, PANEL_W, GAME_BOT - GROUND_ABS))
    pygame.draw.rect(screen, GRASS,  (px, GROUND_ABS, PANEL_W, 5))

    # ── Dino ──────────────────────────────────────────────────────────────
    dx = px + env.DINO_X
    dy = GAME_TOP + int(env.dino_y)   # dy = dino feet (y increases downward)
    _draw_dino(screen, dx, dy, color, env.on_ground)

    # Panel border
    pygame.draw.rect(screen, DIVIDER_COL, (px, GAME_TOP, PANEL_W, GAME_H), 1)


def _draw_footer(screen, px, color, env, epsilon):
    """Dark footer strip with run-time stats."""
    footer_color = tuple(max(0, c - 60) for c in color)
    pygame.draw.rect(screen, footer_color, (px, FOOTER_TOP, PANEL_W, FOOTER_H))

    font = _font(12)
    y    = FOOTER_TOP + 8

    speed_txt = f"Speed: {env.speed:.1f}"
    obs_txt   = f"Obstacles: {len(env.obstacles)}"

    screen.blit(font.render(speed_txt, True, WHITE), (px + 8,  y))
    screen.blit(font.render(obs_txt,   True, WHITE), (px + 8,  y + 16))

    if epsilon is not None:
        eps_str = f"ε = {epsilon:.3f}"
        bar_w   = int((1 - epsilon) * (PANEL_W - 20))   # filled = exploiting
        bar_y   = y + 36
        pygame.draw.rect(screen, DIVIDER_COL, (px + 8, bar_y, PANEL_W - 16, 10))
        pygame.draw.rect(screen, WHITE,       (px + 8, bar_y, bar_w,        10))
        screen.blit(font.render(eps_str, True, WHITE), (px + PANEL_W - 75, y))
        lbl = font.render("exploit ←", True, (180, 180, 180))
        screen.blit(lbl, (px + 8, bar_y + 12))

    if env.done:
        dead_font = _font(20, bold=True)
        dead_surf = dead_font.render("✕  DEAD", True, (255, 80, 80))
        screen.blit(dead_surf, dead_surf.get_rect(
            centerx=px + PANEL_W // 2, centery=FOOTER_TOP + FOOTER_H // 2 - 18))


# ─── Shape drawers ────────────────────────────────────────────────────────────

def _draw_dino(screen, cx, cy, color, on_ground):
    """
    Draw a simple pixel-art dino.
    cx/cy = centre-x, feet-y of the dino.
    """
    W, H = 22, 38
    bx   = cx - W // 2
    by   = cy - H          # top of body

    # Body
    pygame.draw.rect(screen, color, (bx, by, W, H), border_radius=4)

    # Head (slightly lighter, above body)
    hc = tuple(min(255, c + 55) for c in color)
    pygame.draw.rect(screen, hc, (bx + 4, by - 13, 19, 15), border_radius=4)

    # Eye
    pygame.draw.circle(screen, WHITE, (bx + 19, by - 7), 3)
    pygame.draw.circle(screen, BLACK, (bx + 20, by - 7), 1)

    # Tail
    tc = tuple(max(0, c - 35) for c in color)
    pygame.draw.polygon(screen, tc, [
        (bx,      by + H - 6),
        (bx - 10, by + H + 2),
        (bx,      by + H + 2),
    ])

    # Legs — slightly different pose when airborne
    lc = tuple(max(0, c - 20) for c in color)
    if on_ground:
        pygame.draw.rect(screen, lc, (bx + 4,  cy - 9, 6, 9))
        pygame.draw.rect(screen, lc, (bx + 12, cy - 9, 6, 9))
    else:
        # Tuck legs upward while jumping
        pygame.draw.rect(screen, lc, (bx + 4,  cy - 14, 6, 10))
        pygame.draw.rect(screen, lc, (bx + 12, cy - 7,  6,  7))


def _draw_cactus(screen, x, y, w, h):
    """
    Draw a cactus at (x, y) where (x, y) is its top-left corner.
    """
    stem_w = max(7, w // 2)
    stem_x = x + (w - stem_w) // 2

    # Shadow (one pixel right/down)
    pygame.draw.rect(screen, CACTUS_DARK, (stem_x + 1, y + 1, stem_w, h))

    # Main stem
    pygame.draw.rect(screen, CACTUS_MID, (stem_x, y, stem_w, h))

    # Highlight stripe on stem
    pygame.draw.rect(screen, CACTUS_LITE, (stem_x + 2, y + 4, 3, h - 8))

    # Arms only on taller cacti
    if h >= 45:
        arm_base_y  = y + h // 4
        arm_top_y   = arm_base_y - h // 4
        arm_thick   = 5

        # Left arm
        pygame.draw.rect(screen, CACTUS_MID,  (x,          arm_base_y,  stem_x - x + 2, arm_thick))
        pygame.draw.rect(screen, CACTUS_MID,  (x,          arm_top_y,   arm_thick, arm_base_y - arm_top_y + arm_thick))

        # Right arm
        rx = stem_x + stem_w - 2
        pygame.draw.rect(screen, CACTUS_MID,  (rx,         arm_base_y + h // 10, x + w - rx, arm_thick))
        pygame.draw.rect(screen, CACTUS_MID,  (x + w - arm_thick, arm_base_y + h // 10 - h // 5, arm_thick, h // 5 + arm_thick))


# ─── Font cache ───────────────────────────────────────────────────────────────
_font_cache = {}

def _font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("consolas", size, bold=bold)
    return _font_cache[key]
