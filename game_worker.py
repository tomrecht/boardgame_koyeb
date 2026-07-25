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


def worker_play(args):
    """
    Run one game and return (records, winner, score).
    args: (model_state_dict, heuristic_weights, seed, gnn_is_white,
           use_heuristic_opp[, explore_cfg])

    explore_cfg: None = exact legacy greedy behavior (default; the 5-tuple
    form is still accepted so every existing caller is untouched), or
    {'eps': float} -- with probability eps a turn's move-pair is sampled
    UNIFORMLY over the legal scored pairs (EXPLORATION_SPEC.md section 1).
    Rules: the draw call is excluded from sampling (its value is exactly
    known; randomly ending games teaches nothing), a guaranteed win is
    never randomized, and exploration decisions use a DEDICATED RNG stream
    derived from the game seed -- the global `random` dice stream is never
    touched, so a game's dice are identical with exploration on or off.
    Exploratory turns are tagged in the records ('explored': True) for
    diagnostics and possible Watkins-style trace-cutting later.
    """
    if len(args) == 6:
        (model_state_dict, heuristic_weights, seed, gnn_is_white,
         use_heuristic_opp, explore_cfg) = args
    else:
        model_state_dict, heuristic_weights, seed, gnn_is_white, use_heuristic_opp = args
        explore_cfg = None

    # Force CPU-only — must be before any CUDA-touching import path
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import network as _net
    _net.DEVICE = torch.device('cpu')

    from game import Board
    from agent import Agent
    from agent_gnn import GNNAgent
    from network import BoardGNN

    model = BoardGNN()
    model.load_state_dict({k: v.cpu() for k, v in model_state_dict.items()})
    model.eval()
    gnn_agent = GNNAgent(model=model)

    if use_heuristic_opp:
        opp_agent = Agent(weights=heuristic_weights)
    else:
        opp_model = BoardGNN()
        opp_model.load_state_dict({k: v.cpu() for k, v in model_state_dict.items()})
        opp_model.eval()
        opp_agent = GNNAgent(model=opp_model)

    white_agent = gnn_agent if gnn_is_white else opp_agent
    black_agent = opp_agent if gnn_is_white else gnn_agent

    def normalize_chosen(chosen):
        if (isinstance(chosen, tuple) and len(chosen) == 2
                and isinstance(chosen[0], tuple) and len(chosen[0]) == 3):
            return list(chosen)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            return [chosen]
        return list(chosen)

    def serialize_state(board):
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
            })
        return recs

    # Exploration RNG: separate stream, derived from (but not equal to) the
    # game seed, so dice (global `random`, seeded below) are unaffected.
    eps = 0.0
    explore_rng = None
    if explore_cfg and explore_cfg.get('eps', 0) > 0:
        eps = float(explore_cfg['eps'])
        explore_rng = random.Random((seed << 20) ^ 0xE5E5E5)

    random.seed(seed)
    board = Board()
    agents = {'white': white_agent, 'black': black_agent}
    positions = []
    last_total_saved = 0
    turns_since_save = 0

    for turn in range(MAX_TURNS):
        # 1. Normal win
        winner, score = board.check_game_over()
        if winner:
            game_id = f'sp_{seed}_{int(time.time())}'
            return build_records(positions, winner, score, game_id), winner, score

        # 2. No-save draw rule (PRIMARY non-win terminator, owned by the Board).
        # board.draw_callable is set in switch_turn once both players are in
        # midgame and NO_SAVE_TURNS_FOR_DRAW rounds have passed with no save.
        # The trailing side would always claim it, so end as a draw (score 0).
        if board.draw_callable:
            game_id = f'sp_{seed}_draw'
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
            game_id = f'sp_{seed}_stuck'
            return build_records(positions, None, 0, game_id), None, 0

        # 4. Play one turn
        player = board.current_player
        raw_state = serialize_state(board)
        game_stage = board.game_stages.get(player, 'unknown')
        moves = board.get_valid_moves()
        explored_turn = False
        if (explore_rng is not None and explore_rng.random() < eps
                and (not use_heuristic_opp or agents[player] is gnn_agent)):
            # epsilon turn: uniform over legal scored pairs. return_scores
            # costs the same forward pass as the argmax path (it always
            # scores every pair anyway).
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
                          'explored': explored_turn})
        for move in move_list:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()

    # 5. Hard cap reached — unresolved developed position == draw (score 0)
    game_id = f'sp_{seed}_maxturns'
    return build_records(positions, None, 0, game_id), None, 0


# Module-level persistent pool — created once, reused across all iterations.
_POOL = None
_POOL_WORKERS = None


def _pool_worker_init():
    """Runs once per worker at spawn. Guarantees CUDA-free workers."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    # Workers only ever run inference (self-play/eval), never training, so
    # disable autograd process-wide. This skips the VariableType dispatch
    # layer on every tensor op -- notably the per-move board encoding, which
    # runs outside the model's own no_grad blocks and was paying that
    # overhead on all its gather/scatter/view ops.
    torch.set_grad_enabled(False)
    import network as _net
    _net.DEVICE = torch.device('cpu')


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
                             explore_cfg=None):
    global _POOL
    if _POOL is None:
        init_pool(n_workers or 10)

    current_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    args_list = [
        (current_sd, heuristic_weights, seed_offset + i,
         i % 2 == 0, use_heuristic_opp, explore_cfg)
        for i in range(n_games)
    ]

    t0 = time.time()
    records = []
    backstop = 0   # stuck/maxturns backstop terminations (should be rare)
    draws = 0
    done = 0
    print_every = max(1, n_games // 10)

    for recs, winner, score in _POOL.imap_unordered(worker_play, args_list, chunksize=1):
        if recs:
            records.extend(recs)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import network as _net
    _net.DEVICE = torch.device('cpu')
    from game import Board
    from agent import Agent
    from agent_gnn import GNNAgent
    from network import BoardGNN

    ch_model = BoardGNN()
    ch_model.load_state_dict({k: v.cpu() for k, v in challenger_sd.items()})
    ch_model.eval()
    challenger = GNNAgent(model=ch_model)

    if opponent_sd is None:
        opponent = Agent(weights=heuristic_weights)
    else:
        op_model = BoardGNN()
        op_model.load_state_dict({k: v.cpu() for k, v in opponent_sd.items()})
        op_model.eval()
        opponent = GNNAgent(model=op_model)

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
