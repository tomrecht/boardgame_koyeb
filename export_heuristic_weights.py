"""Export the heuristic's EXPANDED weights for the JS port (step 6.2).

Exported, not ported, for the same reason encoder_static.json is: the value the
agent actually runs on is the product of two steps that are easy to get subtly
wrong in a second language --

  1. get_weights() loads best_weights.json and fills EVERY key missing from it
     out of INITIAL_WEIGHTS (the file has no enemy_blot_penalties,
     high_goal_proximity_penalties or permanent_block_bonus, so those come from
     the defaults);
  2. _expand_weights() turns each {a, b} pair into {1..6: a * n**b} plus a 0
     entry, and leaves everything else alone.

A JS reimplementation would have to reproduce both, including which keys are
expanded, and a wrong exponent is a silent scoring difference rather than an
error. Re-run this whenever best_weights.json changes.

    python export_heuristic_weights.py [out.json]
"""
import json
import sys

from agent import Agent, get_weights


def main(out='heuristic_weights.json'):
    # Exactly what app.py hands the served agent.
    raw = get_weights(weights_file='best_weights.json')
    expanded = Agent(weights=raw).weights

    # JSON object keys are strings; the JS side reads them back as strings, so
    # write the numeric per-piece-number keys as strings deliberately.
    def jsonable(v):
        if isinstance(v, dict):
            return {str(k): jsonable(x) for k, x in v.items()}
        return v

    payload = {k: jsonable(v) for k, v in expanded.items()}
    with open(out, 'w') as f:
        json.dump(payload, f, indent=1, sort_keys=True)

    # Self-check: reload and compare against a freshly expanded copy, exactly.
    back = json.load(open(out))
    fresh = Agent(weights=get_weights(weights_file='best_weights.json')).weights
    bad = []
    for key, val in fresh.items():
        got = back.get(key)
        if isinstance(val, dict):
            if set(got) != {str(k) for k in val}:
                bad.append(f'{key}: key set differs')
                continue
            for k, x in val.items():
                if got[str(k)] != x:
                    bad.append(f'{key}[{k}]: {got[str(k)]!r} != {x!r}')
        elif got != val:
            bad.append(f'{key}: {got!r} != {val!r}')
    expanded_cats = [k for k, v in fresh.items()
                     if isinstance(v, dict) and set(v) == {0, 1, 2, 3, 4, 5, 6}]
    print(f'wrote {out}: {len(payload)} keys, {len(expanded_cats)} expanded per-number tables')
    print('self-check:', 'exact' if not bad else f'{len(bad)} MISMATCHES: {bad[:5]}')
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'heuristic_weights.json')
