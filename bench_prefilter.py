"""Speed of the two-stage prefilter vs the current one, same games."""
import random, statistics, sys, time
from game import Board
from agent_gnn import GNNAgent

def run(F, games=6, label=''):
    agent = GNNAgent(weights_path='model.onnx', use_prefilter=True,
                     prefilter_top_k=40, first_move_prefilter=F)
    times = []
    for g in range(games):
        random.seed(100 + g); board = Board()
        for _ in range(120):
            moves = board.get_valid_moves()
            if not moves: break
            t0 = time.perf_counter()
            pair = agent.select_move_pair(moves, board, board.current_player)
            times.append(time.perf_counter() - t0)
            for m in pair:
                if m and m != (0, 0, 0): board.apply_move(m, switch_turn=False)
            board.switch_turn()
            if board.check_game_over()[0]: break
    times.sort(); n = len(times)
    print(f'{label:22} n={n:4}  p50 {times[n//2]:.2f}s  p90 {times[int(.9*n)]:.2f}s  '
          f'p99 {times[min(n-1,int(.99*n))]:.2f}s  max {times[-1]:.2f}s  '
          f'total {sum(times):.0f}s')
    return times

if __name__ == '__main__':
    base = run(0,  label='all pairs (current)')
    for F in (8, 12, 16, 24):
        t = run(F, label=f'first-move top {F}')
        print(f'{"":22} -> {sum(base)/sum(t):.1f}x faster overall, '
              f'{base[-1]/t[-1]:.1f}x on the worst move')
