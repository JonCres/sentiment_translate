# Performance Enhancement Decorators
CACHE_CONFIG = {
    'data_cache_ttl': 3600,  # 1 hour
    'lru_cache_size': 128
}

# Color Palette Constants
COLORS = {
    'critical': '#c62828',
    'warning': '#ef6c00',
    'success': '#2e7d32',
    'primary': '#00a8cc',
    'primary_dark': '#0077aa',
    'text_dark': '#0d1b2a',
    'text_medium': '#495057',
    'text_light': '#868e96',
    'border': '#e9ecef',
    'bg_light': '#f8f9fa',
    'bg_white': '#ffffff',
}

# Shared NPS Colormap for consistent heatmaps (fixed 0-10 scale)
# Uses a continuous red→orange→yellow→lime→green colorscale for NPS visualization
NPS_COLORSCALE = [
    [0.0, '#E53935'],    # 0 - Deep Red (Detractor)
    [0.3, '#FB8C00'],    # 3 - Orange
    [0.5, '#FFEB3B'],    # 5 - Yellow (Neutral/Passive)
    [0.7, '#8BC34A'],    # 7 - Lime Green
    [1.0, '#2E7D32']     # 10 - Deep Green (Promoter)
]
NPS_COLOR_RANGE = [0, 10]  # Fixed min/max for NPS scores
