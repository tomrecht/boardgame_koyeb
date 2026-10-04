"""Build the start-position pool for league_run.py (START_POOL).

Every turn-start position from owner's recorded games against the computer,
after the first MIN_TURN turns (the opening is what ordinary self-play already
covers), replayed exactly through game.py (replay_games' piece-exact
resolution). These are the positions where owner beats the net, so starting
greedy self-play games from them puts training where the net is weakest without
touching the TD targets.

    python3 build_start_pool.py [MIN_TURN=6] quahuru-games-*.jsonl  -> start_pool.jsonl
"""
import json, sys

import weakness_probe as W
from game_worker import serialize_board


def main():
    args = sys.argv[1:]
    min_turn = int(args.pop(0)) if args and args[0].isdigit() else 6
    logs = args or ['quahuru-games-apvo2h65.jsonl', 'quahuru-games-f7in6olg.jsonl']
    seen, out = set(), []
    games = 0
    for f in logs:
        for line in open(f):
            rec = json.loads(line)
            if not rec.get('completed') or rec['id'] in seen:
                continue
            seen.add(rec['id'])
            games += 1
            for t, b, played, b2 in W.turn_states(rec):
                if t < min_turn:
                    continue
                out.append({'game': rec['id'][:8], 'turn': t,
                            'stage': b.get_game_stage(b.current_player),
                            'state': serialize_board(b)})
    with open('start_pool.jsonl', 'w') as fh:
        for r in out:
            fh.write(json.dumps(r) + '\n')
    stages = {}
    for r in out:
        stages[r['stage']] = stages.get(r['stage'], 0) + 1
    print(f'{len(out)} positions from {games} games (turn >= {min_turn}) -> start_pool.jsonl; by stage {stages}')


if __name__ == '__main__':
    main()
