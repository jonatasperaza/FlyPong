"""Painel ao vivo desenhado em pygame (score, raster de atividade neural,
força sinaptica das conexões plasticas ao longo do tempo).

Desenhado diretamente em pygame (em vez de matplotlib) pra nao competir por
thread/backend com o loop do jogo em tempo real.
"""
import numpy as np
import pygame

PANEL_W = 260
RASTER_HISTORY = 150
WEIGHT_HISTORY = 300

FONT_COLOR = (230, 230, 230)
BG_COLOR = (20, 20, 25)
RASTER_COLOR = (80, 200, 255)
WEIGHT_COLOR = (255, 180, 80)


class Dashboard:
    def __init__(self, height, n_raster_neurons):
        self.height = height
        self.n_raster_neurons = n_raster_neurons
        self.raster_buffer = np.zeros((RASTER_HISTORY, n_raster_neurons), dtype=bool)
        self.weight_buffer = []
        self.font = None

    def _ensure_font(self):
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("consolas", 14)

    def push_frame(self, spikes_subset, plastic_weight_mean):
        self.raster_buffer = np.roll(self.raster_buffer, -1, axis=0)
        self.raster_buffer[-1] = spikes_subset
        self.weight_buffer.append(plastic_weight_mean)
        if len(self.weight_buffer) > WEIGHT_HISTORY:
            self.weight_buffer.pop(0)

    def draw(self, surface, x_offset, score_left, score_right, action, dopamine_level):
        self._ensure_font()
        panel = pygame.Rect(x_offset, 0, PANEL_W, self.height)
        pygame.draw.rect(surface, BG_COLOR, panel)

        def text(s, y, color=FONT_COLOR):
            img = self.font.render(s, True, color)
            surface.blit(img, (x_offset + 10, y))

        text(f"Placar: {score_left} x {score_right}", 10)
        text(f"Acao: {'sobe' if action==-1 else 'desce' if action==1 else 'parado'}", 30)
        text(f"Dopamina: {dopamine_level:+.3f}", 50)

        text("Raster (neuronios amostrados):", 80)
        raster_rect = pygame.Rect(x_offset + 10, 100, PANEL_W - 20, 100)
        pygame.draw.rect(surface, (10, 10, 12), raster_rect)
        h, w = self.raster_buffer.shape
        if w > 0:
            cell_w = raster_rect.width / RASTER_HISTORY
            cell_h = raster_rect.height / w
            ys, xs = np.nonzero(self.raster_buffer)
            for t, n in zip(xs, ys):
                px = raster_rect.x + n * cell_w
                py = raster_rect.y + t * cell_h
                pygame.draw.rect(surface, RASTER_COLOR, (px, py, max(1, cell_w), max(1, cell_h)))

        text("Peso sinaptico plastico (media):", 210)
        chart_rect = pygame.Rect(x_offset + 10, 230, PANEL_W - 20, 70)
        pygame.draw.rect(surface, (10, 10, 12), chart_rect)
        if len(self.weight_buffer) > 1:
            wb = np.array(self.weight_buffer)
            wmin, wmax = wb.min(), wb.max()
            span = max(wmax - wmin, 1e-6)
            points = []
            for i, v in enumerate(wb):
                px = chart_rect.x + (i / WEIGHT_HISTORY) * chart_rect.width
                py = chart_rect.y + chart_rect.height - ((v - wmin) / span) * chart_rect.height
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(surface, WEIGHT_COLOR, False, points, 2)
