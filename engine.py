import atexit
import os
import time

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from cache import Cache
import config


class Engine:
    @staticmethod
    def get_engine(engine_path: str):
        """
        Retrieve a chess engine instance using the provided engine path. If the engine is not
        already initialized, a new persistent engine instance is started and stored for future use.

        :param engine_path: Path to the chess engine executable
        :type engine_path: str
        :return: An initialized and persistent chess engine instance
        :rtype: chess.engine.SimpleEngine
        """
        # Yol doğrulaması
        if not engine_path or not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found at path: {engine_path}. Set STOCKFISH_PATH/LC0_PATH correctly.")
        if engine_path not in config._ENGINES:  # Direct access to module variable
            print(f"Starting new persistent engine: {os.path.basename(engine_path)}")
            eng = chess.engine.SimpleEngine.popen_uci(engine_path)
            eng.engine_path = engine_path  # Motorun yolunu nesneye ekle
            # Varsayılan, güvenli yapılandırma (motor-özel)
            base = os.path.basename(engine_path).lower()
            try:
                if "stockfish" in base:
                    # Threads/Hash/MultiPV ortamdan okunabilir
                    threads = int(os.getenv("STOCKFISH_THREADS", max(1, (os.cpu_count() or 1) // 2)))
                    hash_mb = int(os.getenv("STOCKFISH_HASH", 256))
                    opts = {"Threads": threads, "Hash": hash_mb, "MultiPV": int(config.MULTIPV)}
                    Engine._safe_engine_configure(eng, opts)
                elif "lc0" in base:
                    # Lc0 için varsayılanlar (örnekleme modunda noise açık)
                    lc0_opts = {
                        "MultiPV": int(config.LC0_MULTIPV),
                        "Temperature": float(config.LC0_TEMPERATURE),
                        "CPuct": float(config.LC0_CPUCT),
                        "Deterministic": False,
                        "UseNoise": True,
                    }
                    Engine._safe_engine_configure(eng, lc0_opts)
            except Exception:
                # Tüm hataları yut: motorlar farklı opsiyon setlerine sahip olabilir
                pass
            config._ENGINES[engine_path] = eng
        return config._ENGINES[engine_path]


    @staticmethod
    def close_all_engines():
        """
        Closes all persistent engines currently managed by the system.

        This function iterates through a collection of engines and attempts to
        terminate each one. It ensures proper cleanup by clearing the internal
        engine registry after all operations.

        :raises: Any exceptions raised during the termination of engines are
                 caught and suppressed to ensure the process continues for the
                 rest of the engines.
        """
        print("Closing all persistent engines...")
        for engine in config._ENGINES.values():
            try:
                engine.quit()
            except Exception:
                pass
        config._ENGINES.clear()

    # Motorları otomatik kapat
    atexit.register(close_all_engines)

    @staticmethod
    def get_top_moves_and_scores(engine, board, depth=config.TARGET_DEPTH, multipv=config.MULTIPV):
        """
        Analyzes the given board position using the provided chess engine, returning
        the best moves and their associated scores. Now always uses depth-based analysis
        with adaptive nodes for LC0 engines.

        The function interacts with the specified chess engine to extract the top moves
        and their scores based on the analysis depth and the number of variations
        (multipv) to be considered. For LC0 engines, depth is converted to adaptive nodes
        based on position complexity.

        :param engine: The chess engine to be used for analysis.
        :type engine: chess.engine.SimpleEngine
        :param board: The board position to be analyzed.
        :type board: chess.Board
        :param depth: Analysis depth for Stockfish or depth equivalent for LC0. Defaults to TARGET_DEPTH.
        :type depth: int
        :param multipv: The number of principal variations to fetch during analysis.
            Defaults to MULTIPV.
        :type multipv: int
        :return: A tuple containing a list of top moves and their corresponding scores
            as determined by the engine analysis.
        :rtype: tuple[list[chess.Move], list[int]]
        """
        # CACHE EKLENDİ
        cache_key = (getattr(engine, 'engine_path', ''), board.fen(), int(depth), int(multipv))
        if hasattr(config, '_TOP_MOVES_CACHE'):
            cache_dict = config._TOP_MOVES_CACHE
        else:
            config._TOP_MOVES_CACHE = {}
            cache_dict = config._TOP_MOVES_CACHE
        if cache_key in cache_dict:
            print(f"[cache hit] get_top_moves_and_scores: fen='{board.fen()[:20]}...', depth={depth}, multipv={multipv}")
            return cache_dict[cache_key]
        # Yol doğrulaması
        if not getattr(engine, 'engine_path', '') or not os.path.exists(getattr(engine, 'engine_path', '')):
            raise FileNotFoundError(f"Engine not found at path: {getattr(engine, 'engine_path', '')}. Set STOCKFISH_PATH/LC0_PATH correctly.")
        engine_path = getattr(engine, 'engine_path', '')
        is_lc0 = 'lc0' in engine_path.lower()

        if is_lc0:
            # LC0: Convert depth to adaptive nodes
            nodes = config.get_depth_equivalent_nodes(depth)
            limit = chess.engine.Limit(nodes=nodes)
        else:
            # Stockfish: Use depth directly
            limit = chess.engine.Limit(depth=depth)

        info = engine.analyse(board, limit, multipv=multipv)
        moves, scores = [], []
        for i, entry in enumerate(info):
            # Fallback for engines that might not return 'pv' for all multipv entries
            if 'pv' not in entry or not entry['pv']:
                continue
            move = entry['pv'][0]
            score = entry.get('score')
            final_score = -99999 if score is None else score.white().score(mate_score=100000)
            moves.append(move)
            scores.append(final_score)
        cache_dict[cache_key] = (moves, scores)
        return moves, scores

    @staticmethod
    def _safe_engine_configure(engine, options: dict):
        """
        Configures the given engine safely with a set of options. The method attempts to configure
        the engine using all provided options in one go first. If an exception occurs (e.g., due
        to unknown or invalid options), it falls back to configuring the engine on a per-option
        basis to ensure as many valid options as possible are applied without interruptions.

        :param engine: The engine object that is being configured.
        :param options: A dictionary of configuration options to be applied to the engine.
        :type options: dict
        :return: None
        """
        try:
            engine.configure(options)
        except Exception:
            # Try per-option to avoid failure on unknown
            for k, v in (options or {}).items():
                try:
                    engine.configure({k: v})
                except Exception:
                    pass

    @staticmethod
    def lc0_top_moves_and_scores(fen, depth=config.TARGET_DEPTH, multipv=config.LC0_MULTIPV, options=None, time_limit_sec=None, engine=None):
        """
        Analyzes the given chess position using the LC0 chess engine, and retrieves the top moves
        and their corresponding evaluation scores. Now always uses depth-based adaptive nodes system.

        :param fen: FEN string describing the chess position to be analyzed.
        :type fen: str
        :param depth: Analysis depth (converted to adaptive nodes)
        :type depth: int, optional
        :param multipv: Number of top moves to evaluate and return from the analysis.
        :type multipv: int, optional
        :param options: Configuration options for the chess engine.
        :type options: dict, optional
        :param time_limit_sec: Timeout in seconds for the analysis if neither depth is set.
        :type time_limit_sec: float, optional
        :param engine: An instance of the chess engine to use for analysis.
        :type engine: chess.engine.SimpleEngine, optional
        :return: A tuple containing three elements:
                 1. A list of top moves (best moves according to the LC0 engine).
                 2. A list of corresponding move evaluation scores.
                 3. Time elapsed during the analysis in seconds.
        :rtype: tuple[list[chess.Move], list[int], float]
        """
        actual_nodes = config.get_depth_equivalent_nodes(depth)
        cache_key_prefix = f"depth_{depth}_adaptive_{actual_nodes}"

        # Typed-cache lookup (LC0)
        options_key = Cache._make_param_str(options or {})
        cached = Cache.get_typed_cached_result(
            "LC0", "top_moves_scores", expected_type=tuple,
            fen=fen, cache_key=cache_key_prefix,
            multipv=int(multipv), options=options_key, time_limit_sec=time_limit_sec
        )
        if cached is not None:
            return cached

        start = time.perf_counter()
        board = chess.Board(fen)
        if engine is None:
            engine = Engine.get_engine(config.LC0_PATH)

        default_opts = {
            "MultiPV": int(multipv),
            "Temperature": float(config.LC0_TEMPERATURE),
            "CPuct": float(config.LC0_CPUCT),
            "Deterministic": False,
            "UseNoise": True,
        }
        merged_opts = dict(default_opts)
        if options:
            merged_opts.update(options)
        Engine._safe_engine_configure(engine, merged_opts)

        # Analysis with nodes
        try:
            limit = chess.engine.Limit(nodes=int(actual_nodes))
            info = engine.analyse(board, limit, multipv=multipv)
        except Exception:
            # fallback to time
            limit = chess.engine.Limit(time=float(time_limit_sec or 1.0))
            info = engine.analyse(board, limit, multipv=multipv)

        elapsed = time.perf_counter() - start
        moves, scores = [], []
        for entry in info:
            mv = entry.get('pv', [None])[0]
            sc = entry.get('score')
            if mv is None or sc is None:
                continue
            moves.append(mv)
            scores.append(sc.white().score(mate_score=100000) or 100000)

        result = (moves, scores, elapsed)
        Cache.set_typed_cached_result(
            "LC0", "top_moves_scores", result,
            fen=fen, cache_key=cache_key_prefix,
            multipv=int(multipv), options=options_key, time_limit_sec=time_limit_sec
        )
        return result

    @staticmethod
    def lc0_policy_and_moves(fen, depth=None, nodes=None, multipv=config.LC0_MULTIPV, options=None):
        """
        Lc0 motorunun policy head (politika başlığı) katmanından hamleleri ve bu hamlelere karşılık gelen ham olasılıklarını alır.
        Artık adaptif depth sistemi ile çalışır: depth verildiğinde pozisyon karmaşıklığına göre nodes otomatik hesaplanır.

        :param fen: Analiz edilecek pozisyonun FEN (Forsyth-Edwards Notasyonu) formatındaki hali.
        :type fen: str
        :param depth: Analiz derinliği (adaptif nodes ile çalışır)
        :type depth: int, optional
        :param nodes: Analiz sırasında Lc0 motorunun kullanacağı düğüm (node) sayısı. (depth verilirse önceliklidir)
        :type nodes: int, optional
        :param multipv: Analizden döndürülecek en iyi hamle varyasyonu sayısı (MultiPV).
        :type multipv: int
        :param options: Lc0 motoruna iletilecek ek yapılandırma seçeneklerini içeren bir sözlük.
        :type options: dict, optional
        :return: Üç eleman içeren bir tuple:
                 1. (list[chess.Move]): Lc0 tarafından bulunan en iyi hamlelerin listesi.
                 2. (list[float]): Bu hamlelere karşılık gelen normalize edilmiş policy olasılıklarının listesi.
                 3. (float): Analizin ne kadar sürdüğünü saniye cinsinden gösteren süre.
        :rtype: tuple[list[chess.Move], list[float], float]
        """
        # Adaptif nodes hesaplama
        if depth is not None:
            actual_nodes = config.get_depth_equivalent_nodes(depth)
        elif nodes is not None:
            actual_nodes = int(nodes)
        else:
            actual_nodes = int(config.LC0_NODES)

        # Cache kontrolü
        options_key = Cache._make_param_str(options or {})
        cached = Cache.get_typed_cached_result(
            "LC0", "policy_moves", expected_type=tuple,
            fen=fen, nodes=int(actual_nodes), multipv=int(multipv), options=options_key
        )
        if cached is not None:
            return cached

        start = time.perf_counter()
        board = chess.Board(fen)
        engine = Engine.get_engine(config.LC0_PATH)

        # Lc0'dan policy bilgisi almak için özel seçenekler (sadece bir kez configure)
        policy_opts = {
            "MultiPV": int(multipv),
            "Temperature": 0.0,  # Raw policy için
            "VerboseMoveStats": True,  # Policy bilgisi için
            "UCI_ShowWDL": False,
        }
        if options:
            policy_opts.update(options)

        Engine._safe_engine_configure(engine, policy_opts)

        try:
            limit = chess.engine.Limit(nodes=int(actual_nodes))
            info = engine.analyse(board, limit, multipv=multipv)
        except Exception:
            # Fallback to time-limited analysis once
            limit = chess.engine.Limit(time=1.0)
            info = engine.analyse(board, limit, multipv=multipv)

        elapsed = time.perf_counter() - start

        # Collect moves and their ucis
        moves = []
        move_ucis = []
        for entry in info:
            mv = entry.get('pv', [None])[0]
            if mv is None:
                continue
            moves.append(mv)
            mv_uci = mv.uci() if hasattr(mv, 'uci') else str(mv)
            move_ucis.append(mv_uci)

        # If no moves found, return quickly
        if not moves:
            result = ([], [], elapsed)
            Cache.set_typed_cached_result(
                "LC0", "policy_moves", result,
                fen=fen, nodes=int(actual_nodes), multipv=int(multipv), options=options_key
            )
            return result

        # Try to extract numeric policy-like values from the info in a single pass
        # Priority: explicit move_stats/policy dicts > per-entry numeric fields (visits/n) > fallback softmax from scores
        numeric_vals = []  # parallel to move_ucis if available
        for entry, mv_uci in zip(info, move_ucis):
            val = None
            # Check for dict-like move_stats where key is move uci
            for key in ('move_stats', 'moveStats', 'pv_stats', 'policy'):
                stats = entry.get(key)
                if isinstance(stats, dict):
                    # direct lookup with None-safe conversion
                    try:
                        if mv_uci in stats:
                            v = stats.get(mv_uci)
                            if v is not None:
                                val = float(v)
                                break
                    except Exception:
                        pass
                # numeric stat at this key
                if isinstance(stats, (int, float)) and val is None:
                    try:
                        val = float(stats)
                        break
                    except Exception:
                        pass

            # fallback to visits/n/nodes fields
            if val is None:
                for key in ('visits', 'n', 'nodes'):
                    v = entry.get(key)
                    if isinstance(v, (int, float)):
                        try:
                            val = float(v)
                            break
                        except Exception:
                            pass

            numeric_vals.append(val)

        # If we found at least one numeric value, use them (missing entries get small epsilon)
        policy_probs = []
        if any(v is not None for v in numeric_vals):
            arr = np.array([v if v is not None else 0.0 for v in numeric_vals], dtype=np.float64)
            # If all zeros, fallback to uniform
            if arr.sum() <= 0.0:
                policy_probs = [1.0 / len(arr)] * len(arr)
            else:
                # Normalize
                policy_probs = (arr / arr.sum()).tolist()
        else:
            # Single fallback: call top_moves_and_scores once and compute softmax
            try:
                top_moves, top_scores, _ = Engine.lc0_top_moves_and_scores(fen, depth=depth, multipv=multipv, engine=engine)
                # build mapping from move uci to softmax probability
                top_ucis = [tm.uci() if hasattr(tm, 'uci') else str(tm) for tm in top_moves]
                # Convert scores to floats, handle None
                sc = np.array([float(s) if s is not None else -1e6 for s in top_scores], dtype=np.float64)
                # use lambda from config if available for softening
                lam = getattr(config, 'LC0_SOFTMAX_LAMBDA', 1.0) or 1.0
                sc = sc - sc.max()
                # softmax with temperature-like scaling (divide by lam)
                with np.errstate(over='ignore'):
                    soft = np.exp(sc / float(lam))
                if soft.sum() <= 0.0:
                    soft = np.ones_like(soft) / len(soft)
                else:
                    soft = soft / soft.sum()
                prob_map = {u: float(p) for u, p in zip(top_ucis, soft)}
                # assign probabilities to the moves we originally collected
                policy_probs = [prob_map.get(u, 0.0) for u in move_ucis]
                # if all zeros, fall back to uniform
                if sum(policy_probs) <= 0.0:
                    policy_probs = [1.0 / len(move_ucis)] * len(move_ucis)
            except Exception:
                policy_probs = [1.0 / len(move_ucis)] * len(move_ucis)

        # Final normalization with small eps safety
        total = sum(policy_probs)
        if total <= 0.0:
            policy_probs = [1.0 / len(policy_probs)] * len(policy_probs)
        else:
            eps = 1e-12
            policy_probs = [max(p, eps) for p in policy_probs]
            s = sum(policy_probs)
            policy_probs = [p / s for p in policy_probs]

        result = (moves, policy_probs, elapsed)
        Cache.set_typed_cached_result(
            "LC0", "policy_moves", result,
            fen=fen, nodes=int(actual_nodes), multipv=int(multipv), options=options_key
        )
        return result

    @staticmethod
    def sample_paths(fen, depth=None, lam=1.0, samples=1, multipv=None, options=None, use_cache: bool = False, engine=None, mode="competitive", reward_mode="hybrid"):
        """
        Generates sample paths using sequential processing optimized for single GPU systems.
        Uses depth-based adaptive nodes calculation for fair engine comparison.

        Parameters:
        - fen: start position in FEN
        - depth: number of plies to sample for each path and analysis depth per move
        - lam: softmax temperature for move selection
        - samples: number of paths to sample
        - use_cache: enable simple in-memory cache
        - engine: optional pre-initialized engine instance
        - mode: "competitive" veya "quantum_limit" (motor davranışını belirler)
        - reward_mode: "policy", "cp_score", "hybrid" (sampling ödül mekanizması)
        """
        # engine selection
        paths = []
        if engine is None:
            engine = Engine.get_engine(config.LC0_PATH)
            motor_name = "Path-Integral-LC0"
        else:
            engine_path = getattr(engine, 'engine_path', '')
            if 'lc0' in engine_path.lower():
                motor_name = "Path-Integral-LC0"
            elif 'stockfish' in engine_path.lower():
                motor_name = "Path-Integral-Stockfish"
            else:
                motor_name = "Path-Integral-Unknown"

        # choose defaults if not provided
        if motor_name == "Path-Integral-LC0":
            if multipv is None:
                multipv = config.HIGH_MULTIPV if mode == "quantum_limit" else config.COMPETITIVE_MULTIPV
            if depth is None:
                depth = config.HIGH_DEPTH if mode == "quantum_limit" else config.COMPETITIVE_DEPTH
            lc0_opts = {
                "MultiPV": int(multipv),
                "Temperature": 0.0,
                "CPuct": float(config.LC0_CPUCT),
            }
            # merge user options if provided
            merged_opts = dict(lc0_opts)
            if isinstance(options, dict):
                merged_opts.update(options)
            Engine._safe_engine_configure(engine, merged_opts)
            nodes = config.get_depth_equivalent_nodes(depth)
            mode_tag = "quantum_limit" if mode == "quantum_limit" else "competitive"
        else:
            if multipv is None:
                multipv = config.MULTIPV
            if depth is None:
                depth = config.TARGET_DEPTH
            nodes = config.LC0_NODES
            mode_tag = "stockfish"

        # cache key after resolution
        options_key = Cache._make_param_str(options or {})
        cache_key = (fen, int(depth), float(lam), int(samples), mode_tag, int(multipv), options_key)
        if use_cache and cache_key in config._SAMPLE_PATHS_CACHE:
            print(f"[cache hit] sample_paths fen='{fen[:20]}...' depth={depth} λ={lam} n={samples} mode={mode_tag} multipv={multipv}")
            return config._SAMPLE_PATHS_CACHE[cache_key]

        print(f"[DEBUG] sample_paths started: Motor={motor_name}, mode={mode_tag}, depth={depth}, multipv={multipv}, nodes={nodes}, λ={lam}, samples={samples}, options={options_key}")

        tqdm_desc = f"Path-Integral λ={lam} | Motor={motor_name} | mode={mode_tag} | depth={depth} | multipv={multipv} | nodes={nodes} | adapted nodes={config.get_depth_equivalent_nodes(depth)}"
        start_time = time.perf_counter()

        for _ in tqdm(range(samples), desc=tqdm_desc, unit="sample"):
            path = Engine._generate_single_path_depth(fen, depth, lam, engine, mode=mode_tag, multipv=multipv, options=options, reward_mode=reward_mode)
            if path:
                paths.append(path)

        elapsed_time = time.perf_counter() - start_time
        if use_cache:
            config._SAMPLE_PATHS_CACHE[cache_key] = paths

        import gc
        gc.collect()
        print(f"[DEBUG] sample_paths completed: Motor={motor_name}, mode={mode_tag}, paths={len(paths)}, elapsed={elapsed_time:.2f}s")
        return paths

    @staticmethod
    def _generate_single_path_depth(fen, depth, lam, engine, mode="competitive", multipv=None, options=None, reward_mode="hybrid"):
        """
        Generate a single path with depth-based analysis for fair engine comparison.
        Always uses adaptive nodes system - LC0 converts depth to adaptive nodes based on position complexity.
        'depth' parametresi hem yol uzunluğunu hem analiz derinliğini belirler.
        - mode: "competitive" veya "quantum_limit" (motor davranışını belirler)
        - reward_mode: "policy", "cp_score", "hybrid" (sampling ödül mekanizması)
        """
        try:
            board = chess.Board(fen)
        except Exception:
            return []
        path = []
        engine_path = getattr(engine, 'engine_path', '')
        is_lc0 = 'lc0' in engine_path.lower()
        if multipv is None:
            multipv = config.HIGH_MULTIPV if mode == "quantum_limit" else config.COMPETITIVE_MULTIPV
        for _ply in range(int(depth)):
            if board.is_game_over():
                break
            try:
                if is_lc0 and mode == "quantum_limit":
                    if reward_mode == "policy":
                        moves, policy_probs, _ = Engine.lc0_policy_and_moves(
                            board.fen(),
                            depth=depth,
                            multipv=multipv,
                            options=(options or {"MultiPV": multipv})
                        )
                        probs = np.array(policy_probs if policy_probs is not None else [])
                        if probs.size == 0 or not np.all(np.isfinite(probs)):
                            probs = np.ones(len(moves), dtype=np.float64) / max(1, len(moves))
                        else:
                            s = probs.sum()
                            if s <= 0 or not np.isfinite(s):
                                probs = np.ones(len(moves), dtype=np.float64) / max(1, len(moves))
                            else:
                                probs = probs / s
                    elif reward_mode == "cp_score":
                        moves, scores, _ = Engine.lc0_top_moves_and_scores(
                            board.fen(),
                            depth=depth,
                            multipv=multipv,
                            engine=engine,
                            options=(options or {"MultiPV": multipv})
                        )
                        from mathfuncs import Calc
                        probs = Calc.softmax(scores, float(lam))
                        s = probs.sum()
                        if s <= 0 or not np.isfinite(s):
                            probs = np.ones(len(moves), dtype=np.float64) / len(moves)
                        else:
                            probs = probs / s
                    elif reward_mode == "hybrid":
                        moves, policy_probs, _ = Engine.lc0_policy_and_moves(
                            board.fen(),
                            depth=depth,
                            multipv=multipv,
                            options=(options or {"MultiPV": multipv})
                        )
                        moves2, scores, _ = Engine.lc0_top_moves_and_scores(
                            board.fen(),
                            depth=depth,
                            multipv=multipv,
                            engine=engine,
                            options=(options or {"MultiPV": multipv})
                        )
                        # Hareketler aynı sırada ise çarp, değilse eşleştir
                        hybrid_probs = []
                        from mathfuncs import Calc
                        score_probs = Calc.softmax(scores, float(lam))
                        for i, mv in enumerate(moves):
                            try:
                                idx2 = moves2.index(mv)
                                hybrid_probs.append(policy_probs[i] * score_probs[idx2])
                            except ValueError:
                                hybrid_probs.append(policy_probs[i])
                        probs = np.array(hybrid_probs)
                        s = probs.sum()
                        if s <= 0 or not np.isfinite(s):
                            probs = np.ones(len(moves), dtype=np.float64) / len(moves)
                        else:
                            probs = probs / s
                    else:
                        # Varsayılan: policy
                        moves, policy_probs, _ = Engine.lc0_policy_and_moves(
                            board.fen(),
                            depth=depth,
                            multipv=multipv,
                            options=(options or {"MultiPV": multipv})
                        )
                        probs = np.array(policy_probs if policy_probs is not None else [])
                        if probs.size == 0 or not np.all(np.isfinite(probs)):
                            probs = np.ones(len(moves), dtype=np.float64) / max(1, len(moves))
                        else:
                            s = probs.sum()
                            if s <= 0 or not np.isfinite(s):
                                probs = np.ones(len(moves), dtype=np.float64) / max(1, len(moves))
                            else:
                                probs = probs / s
                elif is_lc0:
                    moves, scores, _ = Engine.lc0_top_moves_and_scores(
                        board.fen(),
                        depth=depth,
                        multipv=multipv,
                        engine=engine,
                        options=(options or {"MultiPV": multipv})
                    )
                    from mathfuncs import Calc
                    probs = Calc.softmax(scores, float(lam))
                    s = probs.sum()
                    if s <= 0 or not np.isfinite(s):
                        probs = np.ones(len(moves), dtype=np.float64) / len(moves)
                    else:
                        probs = probs / s
                else:
                    # Stockfish için varsayılan analiz
                    moves, scores = Engine.get_top_moves_and_scores(
                        engine,
                        board,
                        depth=depth,
                        multipv=multipv if multipv is not None else config.MULTIPV
                    )
                    from mathfuncs import Calc
                    probs = Calc.softmax(scores, float(lam))
                    s = probs.sum()
                    if s <= 0 or not np.isfinite(s):
                        probs = np.ones(len(moves), dtype=np.float64) / len(moves)
                    else:
                        probs = probs / s
            except Exception as e:
                print(f"[ERROR] Engine analysis failed at ply {_ply}: {e}")
                break
            if not moves:
                break
            # Normalize move types: prefer chess.Move objects. Convert UCI strings to Move when possible.
            normalized_moves = []
            for mv in moves:
                if isinstance(mv, str):
                    try:
                        normalized_moves.append(chess.Move.from_uci(mv))
                    except Exception:
                        normalized_moves.append(mv)
                else:
                    normalized_moves.append(mv)
            moves = normalized_moves
            try:
                idx = int(np.random.choice(len(moves), p=probs))
            except (ValueError, IndexError):
                idx = 0
            if not moves:
                break
            move = moves[idx]
            if move not in board.legal_moves:
                try:
                    move = list(board.legal_moves)[0]
                except IndexError:
                    break
            path.append(move)
            try:
                board.push(move)
            except Exception:
                break
        return path



    @staticmethod
    def get_stockfish_analysis(fen, depth=config.STOCKFISH_DEPTH, multipv=config.MULTIPV):
        """
        Stockfish motoru kullanarak verilen FEN pozisyonunu analiz eder.
        """
        # Cache kontrolü
        cache_key = f"stockfish_{fen}_{depth}_{multipv}"
        if cache_key in config._ANALYSIS_CACHE:
            print(f"[cache hit] Stockfish analysis: fen='{fen[:20]}...', depth={depth}, multipv={multipv}")
            return config._ANALYSIS_CACHE[cache_key]

        print(f"[INFO] Stockfish analizi başlatılıyor: FEN='{fen[:20]}...', depth={depth}, multipv={multipv}")
        print(f"[INFO] Motor yolu: {config.STOCKFISH_PATH}")
        start_time = time.perf_counter()

        try:
            # Stockfish motorunu al
            if not config.STOCKFISH_PATH or not os.path.exists(config.STOCKFISH_PATH):
                print(f"[ERROR] Stockfish engine not found at path '{config.STOCKFISH_PATH}'. Lütfen config dosyasını ve motor yolunu kontrol edin.")
                return {
                    'best_move': None,
                    'moves': [],
                    'scores': [],
                    'elapsed_time': 0.0,
                    'error': 'Stockfish not found'
                }

            print(f"[INFO] Stockfish motoru başlatılıyor...")
            engine = Engine.get_engine(config.STOCKFISH_PATH)
            board = chess.Board(fen)

            # Stockfish için özel ayarlar
            stockfish_opts = {
                "MultiPV": int(multipv),
                "Threads": max(1, (os.cpu_count() or 1) - 4),
                "Hash": 256,
            }
            Engine._safe_engine_configure(engine, stockfish_opts)
            print(f"[INFO] Motor yapılandırıldı: {stockfish_opts}")

            # Analiz yap
            limit = chess.engine.Limit(depth=int(depth))
            print(f"[INFO] Analiz başlatıldı. Derinlik: {depth}, MultiPV: {multipv}")
            info = engine.analyse(board, limit, multipv=multipv)

            elapsed_time = time.perf_counter() - start_time

            moves = []
            scores = []

            # tqdm ile analiz ilerlemesini göster
            print(f"[INFO] Analiz sonuçları işleniyor...")
            for entry in tqdm(info, desc=f"Stockfish MultiPV", unit="pv"):
                pv = entry.get('pv', [])
                if not pv:
                    continue
                move = pv[0]
                score = entry.get('score')
                if move and score:
                    moves.append(str(move))
                    score_cp = score.white().score(mate_score=100000)
                    if score_cp is None:
                        score_cp = 0
                    scores.append(score_cp)

            print(f"[INFO] Analiz tamamlandı. Süre: {elapsed_time:.2f}s, Hamle sayısı: {len(moves)}")
            if moves:
                print(f"[INFO] En iyi hamle: {moves[0]}, Skor: {scores[0]}")
            else:
                print(f"[WARNING] Analiz sonucu boş veya hata oluştu.")

            result = {
                'best_move': moves[0] if moves else None,
                'moves': moves,
                'scores': scores,
                'elapsed_time': elapsed_time,
                'multipv': multipv,
                'depth': depth
            }

            # Cache'e kaydet
            config._ANALYSIS_CACHE[cache_key] = result

            return result

        except Exception as e:
            print(f"[ERROR] Stockfish analizinde hata: {e}\nTeşhis: Motor yolu, dosya izinleri ve motorun elle çalışıp çalışmadığını kontrol edin.")
            elapsed_time = time.perf_counter() - start_time
            return {
                'best_move': None,
                'moves': [],
                'scores': [],
                'elapsed_time': [],
                'error': [],
            }
    @staticmethod
    def stockfish_top_moves_and_scores(fen, depth=config.TARGET_DEPTH, multipv=config.MULTIPV, options=None, time_limit_sec=None, engine=None):
        """
        Stockfish motoru ile verilen FEN pozisyonunda en iyi hamleleri ve skorları döndürür.
        :param fen: FEN string
        :param depth: Analiz derinliği
        :param multipv: Kaç varyasyon dönecek
        :param options: Motor opsiyonları
        :param time_limit_sec: Zaman limiti (opsiyonel)
        :param engine: Motor nesnesi (opsiyonel)
        :return: (moves, scores, elapsed)
        """
        start = time.perf_counter()
        board = chess.Board(fen)
        if engine is None:
            engine = Engine.get_engine(config.STOCKFISH_PATH)
        if options:
            Engine._safe_engine_configure(engine, options)
        limit = chess.engine.Limit(depth=depth)
        info = engine.analyse(board, limit, multipv=multipv)
        moves, scores = [], []
        for entry in info:
            mv = entry.get('pv', [None])[0]
            sc = entry.get('score')
            if mv is None or sc is None:
                continue
            moves.append(mv.uci())
            scores.append(sc.white().score(mate_score=100000) or 100000)
        elapsed = time.perf_counter() - start
        return moves, scores, elapsed
