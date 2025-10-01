import os

# Engine paths - configurable via environment variables
DEFAULT_STOCKFISH_PATH = "/usr/bin/stockfish"
DEFAULT_LC0_PATH = "/home/batuhanacikgoz04/Documents/GitHub/lc0-path-integral-method/buildDir/lc0"

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)
LC0_PATH = os.getenv("LC0_PATH", DEFAULT_LC0_PATH)

# Test positions for analysis
MULTI_FEN = [
    # Standard chess positions
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",  # Italian Game
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",  # Sicilian Defense
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",  # Ruy Lopez
    "2r3k1/pp2rppp/3p1n2/q2P4/3p1Q2/2N5/PPP2PPP/1K1RR3 w - - 0 19",  # Isolated Pawn Middlegame
    "r3r1k1/pp1n1ppp/2p2q2/3p1b2/3P4/2N1PN2/PP3PPP/R2Q1RK1 w - - 0 13",  # Strategic Middlegame
    "r1b2rk1/pp3pbp/1n1p1np1/2pP4/4P3/2N2N1P/PP2BPP1/R1Bq1RK1 w - - 0 12",  # Tactical Middlegame
    "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",  # King and Pawn Endgame
    # Chess960 positions
    "rbqknrnb/pppppppp/8/8/8/8/PPPPPPPP/RBQKNRNB w KQkq - 0 1",  # Chess960 #0
    "brnqknrb/pppppppp/8/8/8/8/PPPPPPPP/BRNQKNRB w KQkq - 0 1",  # Chess960 #100
    "nbqkrbnr/pppppppp/8/8/8/8/PPPPPPPP/NBQKRBNR w KQkq - 0 1",  # Chess960 #300
    "brnkqrbn/pppppppp/8/8/8/8/PPPPPPPP/BRNKQRBN w KQkq - 0 1",  # Chess960 #700
    "rbnkqbrn/pppppppp/8/8/8/8/PPPPPPPP/RBNKQBRN w KQkq - 0 1",  # Chess960 #900
]

# Chess960 variants with descriptions
CHESS960_VARIANTS = {
    "rbqknrnb/pppppppp/8/8/8/8/PPPPPPPP/RBQKNRNB w KQkq - 0 1": {"variant": 0, "desc": "Chess960 #0 - Bishops on different colors"},
    "brnqknrb/pppppppp/8/8/8/8/PPPPPPPP/BRNQKNRB w KQkq - 0 1": {"variant": 100, "desc": "Chess960 #100 - Bishop and queen adjacent"},
    "nbqkrbnr/pppppppp/8/8/8/8/PPPPPPPP/NBQKRBNR w KQkq - 0 1": {"variant": 300, "desc": "Chess960 #300 - Queen and king in center"},
    "brnkqrbn/pppppppp/8/8/8/8/PPPPPPPP/BRNKQRBN w KQkq - 0 1": {"variant": 700, "desc": "Chess960 #700 - King and queen adjacent"},
    "rbnkqbrn/pppppppp/8/8/8/8/PPPPPPPP/RBNKQBRN w KQkq - 0 1": {"variant": 900, "desc": "Chess960 #900 - King in center"},
}

# Chess960 middlegame positions (derived from opening positions)
CHESS960_MIDGAME_VARIANTS = {
    "rbqknrnb/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/RBQKNRNB w KQkq - 10 10": {"variant": 0, "desc": "Chess960 #0 middlegame"},
    "brnqknrb/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/BRNQKNRB w KQkq - 10 10": {"variant": 100, "desc": "Chess960 #100 middlegame"},
    "nbqkrbnr/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/NBQKRBNR w KQkq - 10 10": {"variant": 300, "desc": "Chess960 #300 middlegame"},
    "brnkqrbn/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/BRNKQRBN w KQkq - 10 10": {"variant": 700, "desc": "Chess960 #700 middlegame"},
    "rbnkqbrn/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/RBNKQBRN w KQkq - 10 10": {"variant": 900, "desc": "Chess960 #900 middlegame"},
}

# Special positions for horizon effect analysis
HORIZON_EFFECT_FENS = {
    "Queen_Sac_Trap": "q6k/5p1p/5P2/8/8/8/8/K7 w - - 0 1",
    "Pawn_Breakthrough": "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"
}

# Default analysis parameters
FEN = MULTI_FEN[0]  # Default position for single-position analysis
SAMPLE_COUNT = 30   # Number of paths to sample per analysis

# Engine analysis parameters
LC0_NODES = 10000        # LC0 nodes budget for analysis
TARGET_DEPTH = 5         # Standard analysis depth
STOCKFISH_DEPTH = 20     # Stockfish analysis depth

# Adaptive nodes system
ADAPTIVE_NODES_ENABLED = True
BASE_NODES_PER_DEPTH = 5000
MIN_NODES_PER_PLY = 100
MAX_NODES_PER_PLY = 1000000

# Path integral framework parameters
LAMBDA = 0.18             # Default lambda for path integral framework
MULTIPV = 5              # Number of top moves to analyze
TOP_N = 5                # Top N moves for accuracy calculation

# Analysis ranges for sensitivity studies
LAMBDA_SCAN = [0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
               0.12, 0.15, 0.18, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0]
SAMPLE_SIZES = [50, 100, 200, 500]
DEPTH_SCAN = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30]

# LC0 specific parameters
LC0_MULTIPV = 5
LC0_TEMPERATURE = 0.7
LC0_CPUCT = 1.0
LC0_SOFTMAX_LAMBDA = 0.7

# Performance and memory management
SKIP_EXISTING_RESULTS = True
MEMORY_CLEANUP_INTERVAL = 2
MAX_MEMORY_USAGE_MB = 20480

# Global caches (managed by cache.py)
_ENGINES = {}
_SAMPLE_PATHS_CACHE: dict[tuple, list] = {}
_ANALYSIS_CACHE: dict[str, dict] = {}

# Utility functions for cache and engine management
def get_engines():
    """Get the global engines cache"""
    return _ENGINES

def get_sample_cache():
    """Get the global sample paths cache"""
    return _SAMPLE_PATHS_CACHE

def get_analysis_cache():
    """Get the global analysis cache"""
    return _ANALYSIS_CACHE

# Adaptive nodes calculation
def calculate_adaptive_nodes(depth):
    """Calculate adaptive nodes based on depth"""
    if not ADAPTIVE_NODES_ENABLED:
        return LC0_NODES
    
    base_nodes = BASE_NODES_PER_DEPTH * depth
    adaptive_nodes = int(base_nodes * 1.3)  # Complexity multiplier
    
    return max(MIN_NODES_PER_PLY, min(MAX_NODES_PER_PLY, adaptive_nodes))

def get_depth_equivalent_nodes(depth):
    """Get LC0 nodes equivalent to Stockfish depth for fair comparison"""
    return calculate_adaptive_nodes(depth)

# Analysis modes
MODE = os.getenv("PI_MODE", "competitive")  # "competitive" or "quantum_limit"

# Mode-specific parameters
HIGH_MULTIPV = 20        # MultiPV for quantum limit mode
HIGH_DEPTH = 10          # Depth for quantum limit mode
COMPETITIVE_MULTIPV = 5  # MultiPV for competitive mode
COMPETITIVE_DEPTH = 5      # Depth for competitive mode