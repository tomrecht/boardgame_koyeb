"""
game_worker.py — Worker process for parallel game generation.
CPU-only inference in workers; GPU reserved for training in main process.

Game termination (all non-win exits score as a DRAW, final_score 0, to match
the real game rules):
  1. Normal win (check_game_over).
  2. No-save draw rule: once both players are in midgame, if NO_SAVE_TURNS_FOR_DRAW
     full rounds pass with no save, either player may call a draw. In self-play
     the trailing side always would (a draw is valued at 0, so it's only
     declined by a side that expects to do better), so we terminate
     deterministically as a draw the moment board.draw_callable is set. This is
     the PRIMARY non-win terminator and is owned by the Board.
  3. STUCK_LIMIT: loose backstop (set above the draw threshold) for the rare
     case the draw gate never arms (e.g. a side still has unentered pieces).
     Scored as a draw.
  4. MAX_TURNS: hard cap, scored as a draw. (Games can't time out in the
     opening because both sides must bring a piece out every turn, so any
     timeout is a developed-but-unresolved position == a draw.)
Partial positions are always recorded (even on draw/stuck/timeout) so no
game is wasted.
"""
import random, time, os, math
import torch
import multiprocessing as mp

from game import NO_SAVE_TURNS_FOR_DRAW

MAX_TURNS   = 200
# Loose backstop only. The no-save draw rule fires at NO_SAVE_TURNS_FOR_DRAW
# ROUNDS (= 2x player-turns) once both sides are in midgame, so it will
# essentially always trigger before this player-turn counter does. Kept above
# that effective threshold so it only catches games where the draw gate never
# arms.
STUCK_LIMIT = 2 * NO_SAVE_TURNS_FOR_DRAW + 10


_OPP_CACHE = {}   # per-worker: checkpoint path -> GNNAgent (league opponents)


def _set_worker_device():
    """The device a worker runs inference on: $WORKER_DEVICE ('cpu' default,
    'cuda' on a GPU box, 'mps' on a Mac). Training in the main process follows
    $BOARDGAME_DEVICE separately. Workers stay CUDA-free unless asked."""
    import torch
    import network as _net
    dev = os.environ.get('WORKER_DEVICE', 'cpu')
    if dev == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _net.DEVICE = torch.device(dev)
    return _net.DEVICE


def serialize_board(board):
    """A position in the frontend's state format (Board.update_state reads it)."""
    return {
        'currentTurn': board.current_player,
        'dice': [{'value': d.number, 'used': d.used} for d in board.dice],
        'racks': {
            'whiteUnentered': [{'color': 'white', 'number': p.number} for p in board.white_unentered],
            'whiteSaved':     [{'color': 'white', 'number': p.number} for p in board.white_saved],
            'blackUnentered': [{'color': 'black', 'number': p.number} for p in board.black_unentered],
            'blackSaved':     [{'color': 'black', 'number': p.number} for p in board.black_saved],
        },
        'boardPieces': [
            {'color': p.player, 'number': p.number,
             'tile': {'ring': p.tile.ring, 'sector': p.tile.pos}}
            for p in board.pieces if p.tile is not None
        ],
    }


def _league_opponent(path, hand_rules):
    """Load (once per worker process) a frozen panel member as a GNNAgent."""
    key = (path, bool(hand_rules))
    if key not in _OPP_CACHE:
        from network import model_from_state
        from agent_gnn import GNNAgent
        m = model_from_state(torch.load(path, map_location='cpu'))
        _OPP_CACHE[key] = GNNAgent(model=m, hand_rules=hand_rules)
    return _OPP_CACHE[key]


def worker_play(args):
    """
    Run one game and return (records, winner, score).
    args: (model_state_dict, heuristic_weights, seed, gnn_is_white,
           use_heuristic_opp[, cfg])

    cfg: None = exact legacy greedy self-play (the 5-tuple form is still
    accepted so every existing caller is untouched), or a dict with any of:
      'eps'           legacy epsilon-greedy: with probability eps a turn's
                      pair is sampled UNIFORMLY over the legal scored pairs.
      'softmax_T'     softmax OPENING exploration (explore.py): for each
      'softmax_turns' LEARNER side's first softmax_turns turns, sample the pair
                      from softmax(margin / T) over its 2-ply scores.
      'opp_path'      LEAGUE game: the opponent is this frozen checkpoint
      'opp_tag'       instead of the learner itself ('heuristic' as opp_path
                      means the heuristic agent). Opponent always plays greedy.
      'hand_rules'    the agent's hand-coded play rules (default False: the
                      training pipeline measures what the net learned itself).
      'start_state'   START the game from this position (serialize_board
                      format) instead of the opening; dice are re-rolled from
                      the game seed. Opening softmax does not apply in such a
                      game (its first turns are not the opening).
      'late_T'        LATE exploration: on a learner turn past its opening
      'late_p'        window, with probability late_p sample the pair from
                      softmax(margin / late_T). Records are marked 'explored'
                      so TD targets can CUT the trace there (TRACE_CUT=1,
                      td_returns) -- never enable without it.
    Exploration (either kind) only ever applies to LEARNER turns, uses a
    DEDICATED RNG derived from the game seed -- the global `random` dice stream
    is never touched, so a game's dice are identical with exploration on or
    off -- never samples the draw call and always takes a guaranteed win.

    Every record carries 'learner' (True iff the side to move is the network
    being trained; both sides in self-play) and 'opp' (the opponent tag, or
    'self'). Training keeps only learner records -- see td_selfplay_loop.
    """
    if len(args) == 6:
        (model_state_dict, heuristic_weights, seed, gnn_is_white,
         use_heuristic_opp, cfg) = args
    else:
        model_state_dict, heuristic_weights, seed, gnn_is_white, use_heuristic_opp = args
        cfg = None
    cfg = cfg or {}

    # Device from $WORKER_DEVICE (default CPU) -- before any CUDA-touching import
    import torch
    import network as _net
    _set_worker_device()

    from game import Board
    from agent import Agent
    from agent_gnn import GNNAgent, SCORE_SCALE
    from network import BoardGNN, model_from_state
    import explore

    hand_rules = bool(cfg.get('hand_rules', False))
    model = model_from_state(model_state_dict)
    gnn_agent = GNNAgent(model=model, hand_rules=hand_rules)

    opp_path = cfg.get('opp_path')
    league = opp_path is not None
    opp_tag = cfg.get('opp_tag', 'self') if league else ('heuristic' if use_heuristic_opp else 'self')
    if league and opp_path == 'heuristic':
        opp_agent = Agent(weights=heuristic_weights)
    elif league:
        opp_agent = _league_opponent(opp_path, hand_rules)
    elif use_heuristic_opp:
        opp_agent = Agent(weights=heuristic_weights)
    else:
        opp_model = model_from_state(model_state_dict)
        opp_agent = GNNAgent(model=opp_model, hand_rules=hand_rules)

    white_agent = gnn_agent if gnn_is_white else opp_agent
    black_agent = opp_agent if gnn_is_white else gnn_agent
    learner_color = 'white' if gnn_is_white else 'black'

    def is_learner(player):
        # Self-play (opponent = the learner's own weights): both sides learn.
        # League / heuristic games: only the learner's colour.
        if opp_tag == 'self':
            return True
        return player == learner_color

    def normalize_chosen(chosen):
        if (isinstance(chosen, tuple) and len(chosen) == 2
                and isinstance(chosen[0], tuple) and len(chosen[0]) == 3):
            return list(chosen)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            return [chosen]
        return list(chosen)

    serialize_state = serialize_board

    def build_records(positions, winner, score, game_id):
        recs = []
        total = len(positions)
        for i, pos in enumerate(positions):
            ply = total - i
            player = pos['player']
            if winner:
                final_score = score if player == winner else -score
            else:
                final_score = 0
            recs.append({
                'game_id':      game_id,
                'player':       player,
                'game_stage':   pos['game_stage'],
                'move_index':   pos['move_index'],
                'raw_state':    pos['raw_state'],
                'final_score':  final_score,
                'ply_from_end': ply,
                'explored':     pos.get('explored', False),
                'sampled':      pos.get('sampled', False),
                'explore_gap':  pos.get('explore_gap', 0.0),
                'learner':      is_learner(player),
                'opp':          opp_tag,
                'game_secs':    time.time() - game_t0,   # single-core cost of the game
            })
        return recs

    # Exploration RNG: separate stream, derived from (but not equal to) the
    # game seed, so dice (global `random`, seeded below) are unaffected.
    eps = float(cfg.get('eps', 0) or 0)
    soft_T = float(cfg.get('softmax_T', 0) or 0)
    soft_turns = int(cfg.get('softmax_turns', 0) or 0)
    late_T = float(cfg.get('late_T', 0) or 0)
    late_p = float(cfg.get('late_p', 0) or 0)
    start_state = cfg.get('start_state')
    if start_state is not None:
        soft_T = 0.0                      # a mid-game start has no opening
    explore_rng = None
    if eps > 0 or (soft_T > 0 and soft_turns > 0) or (late_T > 0 and late_p > 0):
        explore_rng = random.Random((seed << 20) ^ 0xE5E5E5)

    random.seed(seed)
    game_t0 = time.time()
    board = Board()
    if start_state is not None:
        board.update_state(start_state)
        board.firstMove = None
        for d in board.dice:              # fresh roll from this game's seed
            d.roll()
    agents = {'white': white_agent, 'black': black_agent}
    positions = []
    last_total_saved = 0
    turns_since_save = 0
    side_turns = {'white': 0, 'black': 0}   # turns played so far by each side

    # game_id carries the seed (unique per generation call) and the opponent,
    # so league games can never collide with self-play ones in the replay pool.
    gid_base = f'sp_{seed}' if opp_tag == 'self' else f'lg_{seed}_{opp_tag}'
    if start_state is not None:
        gid_base += '_st'

    for turn in range(MAX_TURNS):
        # 1. Normal win
        winner, score = board.check_game_over()
        if winner:
            game_id = f'{gid_base}_{int(time.time())}'
            return build_records(positions, winner, score, game_id), winner, score

        # 2. No-save draw rule (PRIMARY non-win terminator, owned by the Board).
        # board.draw_callable is set in switch_turn once both players are in
        # midgame and NO_SAVE_TURNS_FOR_DRAW rounds have passed with no save.
        # The trailing side would always claim it, so end as a draw (score 0).
        if board.draw_callable:
            game_id = f'{gid_base}_draw'
            return build_records(positions, None, 0, game_id), None, 0

        # 3. Stuck backstop — only for the rare case the draw gate never armed
        # (e.g. a side still has unentered pieces). Scored as a draw.
        current_saved = len(board.white_saved) + len(board.black_saved)
        if current_saved > last_total_saved:
            last_total_saved = current_saved
            turns_since_save = 0
        elif last_total_saved > 0:
            turns_since_save += 1

        if last_total_saved > 0 and turns_since_save >= STUCK_LIMIT:
            game_id = f'{gid_base}_stuck'
            return build_records(positions, None, 0, game_id), None, 0

        # 4. Play one turn
        player = board.current_player
        raw_state = serialize_state(board)
        game_stage = board.game_stages.get(player, 'unknown')
        moves = board.get_valid_moves()
        explored_turn = False
        sampled_turn = False
        explore_gap = 0.0
        learner_turn = is_learner(player)
        if (explore_rng is not None and learner_turn and soft_T > 0
                and side_turns[player] < soft_turns):
            # Softmax opening exploration (explore.py). return_scores costs the
            # same forward pass as the argmax path.
            ranked = agents[player].select_move_pair(moves, board, player,
                                                     return_scores=True)
            if isinstance(ranked, tuple):            # defensive: bare pair
                chosen = ranked
            else:
                chosen, explore_gap = explore.softmax_pick(
                    ranked, soft_T, explore_rng, SCORE_SCALE)
                explored_turn = explore_gap > 1e-6
                chosen = gnn_agent._dedupe_save_pair(chosen)
                sampled_turn = True
        elif (explore_rng is not None and late_T > 0 and learner_turn
              and side_turns[player] >= soft_turns
              and explore_rng.random() < late_p):
            # Late exploration (requires TRACE_CUT=1 in training, see docstring).
            ranked = agents[player].select_move_pair(moves, board, player,
                                                     return_scores=True)
            if isinstance(ranked, tuple):            # defensive: bare pair
                chosen = ranked
            else:
                chosen, explore_gap = explore.softmax_pick(
                    ranked, late_T, explore_rng, SCORE_SCALE)
                explored_turn = explore_gap > 1e-6
                chosen = gnn_agent._dedupe_save_pair(chosen)
                sampled_turn = True
        elif (explore_rng is not None and eps > 0 and learner_turn
              and explore_rng.random() < eps):
            # legacy epsilon turn: uniform over legal scored pairs.
            ranked = agents[player].select_move_pair(moves, board, player,
                                                     return_scores=True)
            if isinstance(ranked, tuple):            # defensive: bare pair
                chosen = ranked
            elif ranked[0][0] == float('inf'):       # guaranteed win: take it
                chosen = ranked[0][1]
            else:
                draw_pair = ((1, 1, 1), (0, 0, 0))
                pairs = [p for _s, p in ranked if p != draw_pair]
                if pairs:
                    chosen = explore_rng.choice(pairs)
                    explored_turn = len(pairs) > 1   # a forced move isn't exploration
                else:
                    chosen = draw_pair
        else:
            chosen = agents[player].select_move_pair(moves, board, player)

        move_list = normalize_chosen(chosen)
        positions.append({'player': player, 'game_stage': game_stage,
                          'move_index': turn, 'raw_state': raw_state,
                          'explored': explored_turn, 'sampled': sampled_turn,
                          'explore_gap': explore_gap})
        side_turns[player] += 1
        for move in move_list:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()

    # 5. Hard cap reached — unresolved developed position == draw (score 0)
    game_id = f'{gid_base}_maxturns'
    return build_records(positions, None, 0, game_id), None, 0


# Module-level persistent pool — created once, reused across all iterations.
_POOL = None
_POOL_WORKERS = None


def _pool_worker_init():
    """Runs once per worker at spawn. CUDA-free unless $WORKER_DEVICE says so."""
    if os.environ.get('WORKER_DEVICE', 'cpu') == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    # Workers only ever run inference (self-play/eval), never training, so
    # disable autograd process-wide. This skips the VariableType dispatch
    # layer on every tensor op -- notably the per-move board encoding, which
    # runs outside the model's own no_grad blocks and was paying that
    # overhead on all its gather/scatter/view ops.
    torch.set_grad_enabled(False)
    _set_worker_device()


def init_pool(n_workers=5):
    """
    Create the persistent worker pool ONCE. Call this in its own cell,
    BEFORE the training loop. Uses spawn so workers never inherit the
    parent's CUDA context (which is what was deadlocking fork/forkserver).
    Startup cost (~1-2 min for PyTorch import per worker) is paid once here.
    """
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        print(f"Pool already running with {_POOL_WORKERS} workers.")
        return
    # Ensure children inherit a CUDA-free environment from the parent too
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ctx = mp.get_context('spawn')
    print(f"Starting {n_workers} workers (one-time, ~1-2 min)...")
    t0 = time.time()
    _POOL = ctx.Pool(processes=n_workers, initializer=_pool_worker_init)
    # Warm up: force all workers to actually start by running trivial tasks
    _POOL.map(_noop, range(n_workers))
    _POOL_WORKERS = n_workers
    print(f"Pool ready in {time.time()-t0:.0f}s.")


def _noop(_):
    return True


def shutdown_pool():
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
        _POOL_WORKERS = None
        print("Pool shut down.")


def generate_games_parallel(model, opp_state_dict_or_none,
                             n_games, seed_offset,
                             heuristic_weights,
                             use_heuristic_opp=False,
                             n_workers=None,  # ignored if pool already running
                             label='',
                             explore_cfg=None,
                             league_cfg=None,
                             start_cfg=None):
    """Generate n_games of training games on the pool.

    explore_cfg: per-game cfg dict passed to worker_play (exploration keys,
      'hand_rules'); None = legacy greedy self-play.
    league_cfg: None = all self-play, or
      {'frac': f, 'opponents': {tag: path_or_'heuristic'}, 'seed': s}
      -- a fraction f of games is played against an opponent drawn uniformly
      from `opponents`. Which games are league games, and against whom, is a
      pure function of (seed, game index), so a resumed iteration reproduces it.
      The learner's colour alternates by game index exactly as in self-play.
    start_cfg: None, or {'frac': f, 'pool': [state, ...], 'seed': s,
      'rotate': bool} -- a fraction f of games START from a position drawn from
      `pool` (serialize_board format; see build_start_pool.py), optionally
      rotated by a random one of the board's 3 symmetric images. Drawn on its
      own RNG stream, so the league schedule is unchanged by it.
    Returns the record list; `generate_games_parallel.last_counts` holds
    {opp_tag: games} for the call (asserted on by the smoke test)."""
    global _POOL
    if _POOL is None:
        init_pool(n_workers or 10)

    current_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    base_cfg = dict(explore_cfg or {})
    opps = sorted((league_cfg or {}).get('opponents', {}).items())
    frac = float((league_cfg or {}).get('frac', 0.0)) if opps else 0.0
    sched_rng = random.Random(((league_cfg or {}).get('seed', 0) << 16) ^ seed_offset)
    args_list = []
    planned = {}
    pool = (start_cfg or {}).get('pool') or []
    s_frac = float((start_cfg or {}).get('frac', 0.0)) if pool else 0.0
    start_rng = random.Random((((start_cfg or {}).get('seed', 0) << 16) ^ seed_offset) ^ 0x5157)
    sym = None
    if s_frac > 0 and (start_cfg or {}).get('rotate'):
        from symmetry import Symmetry
        sym = Symmetry()
    n_starts = 0
    for i in range(n_games):
        cfg = dict(base_cfg)
        if frac > 0 and sched_rng.random() < frac:
            tag, path = opps[sched_rng.randrange(len(opps))]
            cfg['opp_tag'], cfg['opp_path'] = tag, path
        if s_frac > 0 and start_rng.random() < s_frac:
            st = pool[start_rng.randrange(len(pool))]
            if sym is not None:
                st = sym.transform(st, start_rng.randrange(3))
            cfg['start_state'] = st
            n_starts += 1
        planned[cfg.get('opp_tag', 'self')] = planned.get(cfg.get('opp_tag', 'self'), 0) + 1
        args_list.append((current_sd, heuristic_weights, seed_offset + i,
                          i % 2 == 0, use_heuristic_opp, cfg or None))

    if s_frac > 0:
        print(f'  {label}: {n_starts} of {n_games} games start from a pool position '
              f'(pool {len(pool)}, rotate {sym is not None})')
    t0 = time.time()
    records = []
    backstop = 0   # stuck/maxturns backstop terminations (should be rare)
    draws = 0
    done = 0
    print_every = max(1, n_games // 10)

    played = {}       # opp_tag -> games actually returned (counted from records)
    learner_wins = {}  # opp_tag -> learner wins (league games only meaningful)
    secs_by_kind = {}  # 'self' / 'league' -> single-core seconds per game
    for recs, winner, score in _POOL.imap_unordered(worker_play, args_list, chunksize=1):
        if recs:
            records.extend(recs)
            tag = recs[0].get('opp', 'self')
            played[tag] = played.get(tag, 0) + 1
            kind = 'self' if tag == 'self' else 'league'
            secs_by_kind.setdefault(kind, []).append(recs[0]['game_secs'])
            if tag != 'self' and winner is not None:
                lw = any(r['learner'] and r['player'] == winner for r in recs)
                learner_wins[tag] = learner_wins.get(tag, 0) + int(lw)
        if winner is None:
            draws += 1
        if recs and recs[0]['game_id'].endswith(('stuck', 'maxturns')):
            backstop += 1
        done += 1
        if done % print_every == 0 or done == n_games:
            print(f'  {label}: {done}/{n_games} games ({time.time()-t0:.0f}s, '
                  f'{len(records)} positions, {draws} draws, {backstop} backstop)')

    elapsed = time.time() - t0
    print(f'  {label}: {n_games} games ({draws} draws, of which {backstop} backstop), '
          f'{len(records)} positions, {elapsed:.0f}s '
          f'({elapsed/n_games:.1f}s/game, {_POOL_WORKERS} workers)')
    if any(t != 'self' for t in planned):
        print(f'  {label}: opponents planned {planned} | played {played} | '
              f'learner wins vs league {learner_wins}')
    sampled = sum(1 for r in records if r.get('sampled'))
    if sampled:
        explored = sum(1 for r in records if r.get('explored'))
        gaps = [r['explore_gap'] for r in records if r.get('sampled')]
        last = max(r['move_index'] for r in records if r.get('sampled'))
        opp_sampled = sum(1 for r in records if r.get('sampled') and not r['learner'])
        print(f'  {label}: softmax-sampled turns {sampled}, of which '
              f'{explored} chose a pair the net rates strictly worse than its '
              f'best (mean gap {sum(gaps) / len(gaps):.3f}, max {max(gaps):.3f} '
              f'margin points); latest sampled game turn '
              f'{last}; sampled on a league opponent\'s turn {opp_sampled}')
        assert opp_sampled == 0, 'exploration fired on a panel opponent turn'
    print(f'  {label}: single-core s/game ' + ', '.join(
        f'{k} {sum(v) / len(v):.0f}s (n={len(v)})' for k, v in sorted(secs_by_kind.items())))
    generate_games_parallel.last_counts = {'planned': planned, 'played': played,
                                           'learner_wins': learner_wins,
                                           'secs': {k: sum(v) / len(v) for k, v in secs_by_kind.items()}}
    return records


# ============================================================
# Parallel evaluation — challenger GNN vs opponent (GNN or heuristic)
# Runs games on CPU in the persistent pool. Returns win rate.
# Solves both the cuda/cpu device crash and the sequential-eval time sink.
# ============================================================

def worker_eval(args):
    """
    Play one evaluation game. Returns (challenger_won, is_draw).
    args: (challenger_sd, opponent_sd_or_None, heuristic_weights,
           seed, challenger_is_white)
    If opponent_sd_or_None is None -> opponent is the heuristic agent.
    """
    (challenger_sd, opponent_sd, heuristic_weights,
     seed, challenger_is_white) = args

    import torch
    import network as _net
    _set_worker_device()
    from game import Board
    from agent import Agent
    from agent_gnn import GNNAgent
    from network import BoardGNN, model_from_state
    import explore
    samples_before = explore.samples()

    # Hand-coded play rules OFF: gating measures the nets, not the rules.
    ch_model = model_from_state(challenger_sd)
    challenger = GNNAgent(model=ch_model, hand_rules=False)

    if opponent_sd is None:
        opponent = Agent(weights=heuristic_weights)
    else:
        op_model = model_from_state(opponent_sd)
        opponent = GNNAgent(model=op_model, hand_rules=False)

    white = challenger if challenger_is_white else opponent
    black = opponent if challenger_is_white else challenger
    agents = {'white': white, 'black': black}

    def normalize_chosen(chosen):
        if (isinstance(chosen, tuple) and len(chosen) == 2
                and isinstance(chosen[0], tuple) and len(chosen[0]) == 3):
            return list(chosen)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            return [chosen]
        return list(chosen)

    import random
    random.seed(seed)
    board = Board()
    last_saved = 0
    turns_since_save = 0

    winner = None
    for turn in range(MAX_TURNS):
        winner, score = board.check_game_over()
        if winner:
            break
        # No-save draw rule (primary): a draw for evaluation purposes.
        if board.draw_callable:
            winner = None
            break
        # Stuck backstop (draw).
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved = cur
            turns_since_save = 0
        elif last_saved > 0:
            turns_since_save += 1
        if last_saved > 0 and turns_since_save >= STUCK_LIMIT:
            winner = None
            break
        player = board.current_player
        chosen = agents[player].select_move_pair(board.get_valid_moves(), board, player)
        for m in normalize_chosen(chosen):
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    else:
        # MAX_TURNS hit without resolution -> draw
        winner = None

    # Evaluation is strictly greedy: no exploration sample may have been drawn.
    assert explore.samples() == samples_before, 'exploration fired during EVAL'
    if winner is None:
        return (False, True)
    challenger_color = 'white' if challenger_is_white else 'black'
    return (winner == challenger_color, False)


def evaluate_parallel(challenger_model, opponent_sd_or_none,
                      n_games, seed_offset, heuristic_weights,
                      label='Eval', promote_winrate=None,
                      sprt_alpha=0.02, sprt_beta=0.02):
    """
    Win rate of challenger vs opponent over n_games (alternating colors),
    run in parallel on the persistent pool. Draws excluded from win rate
    denominator. Returns win_rate (float).

    If promote_winrate is given, applies a one-sided SPRT futility check
    (H0: p<=promote_winrate-0.05 vs H1: p>=promote_winrate+0.05) after each
    decisive result. Only the lower (futility) boundary is acted on -- a
    result that's actually trending toward promotion always plays out the
    full n_games, so the promote_winrate gate keeps its full noise-robustness
    for any real promotion decision. sprt_beta bounds the probability of
    wrongly cutting off a genuinely-promoting model early; sprt_alpha is the
    complementary (unused-in-practice) false-promote rate that the boundary
    formula also depends on.

    Dispatches in small batches (not all n_games at once) so an early SPRT
    stop actually frees the worker pool quickly -- Pool.imap_unordered
    queues its whole args_list up front regardless of whether the consumer
    keeps reading results, so batching bounds the wasted/abandoned work to
    about one batch instead of the whole remaining eval set.
    """
    global _POOL
    if _POOL is None:
        init_pool(10)

    ch_sd = {k: v.cpu() for k, v in challenger_model.state_dict().items()}

    sprt_enabled = promote_winrate is not None
    if sprt_enabled:
        margin = 0.05
        p0 = promote_winrate - margin
        p1 = promote_winrate + margin
        llr_win = math.log(p1 / p0)
        llr_loss = math.log((1 - p1) / (1 - p0))
        lower_bound = math.log(sprt_beta / (1 - sprt_alpha))
        llr = 0.0

    batch_size = max((_POOL_WORKERS or 5) * 2, 1)

    t0 = time.time()
    wins = 0
    decisive = 0
    draws = 0
    done = 0
    print_every = max(1, n_games // 10)
    stopped_early = False

    next_i = 0
    while next_i < n_games and not stopped_early:
        batch = range(next_i, min(next_i + batch_size, n_games))
        next_i += len(batch)
        # Paired seeds: games (2k, 2k+1) share dice stream seed_offset+k with
        # challenger colors swapped (duplicate-bridge style), cancelling
        # dice-luck variance across the pair. eval seeds therefore span only
        # [seed_offset, seed_offset + n_games//2) -- callers must space
        # seed_offset accordingly.
        args_batch = [(ch_sd, opponent_sd_or_none, heuristic_weights,
                       seed_offset + i // 2, i % 2 == 0) for i in batch]

        for won, is_draw in _POOL.imap_unordered(worker_eval, args_batch, chunksize=1):
            done += 1
            if is_draw:
                draws += 1
            else:
                decisive += 1
                if won:
                    wins += 1
                if sprt_enabled:
                    llr += llr_win if won else llr_loss
            if done % print_every == 0 or done == n_games:
                print(f'  {label}: {done}/{n_games} games ({time.time()-t0:.0f}s, '
                      f'{wins} wins, {draws} draws)')
            if sprt_enabled and decisive >= 10 and llr <= lower_bound:
                stopped_early = True

    if stopped_early:
        print(f'  {label}: SPRT futility stop after {done}/{n_games} games '
              f'({wins}/{decisive} decisive wins so far) -- promotion '
              f'statistically implausible, skipping remaining evals.')

    wr = wins / decisive if decisive else 0.0
    print(f'  {label}: {wins}/{decisive} decisive wins ({draws} draws) -> {wr:.1%}'
          + (' [early stop]' if stopped_early else ''))
    return wr
