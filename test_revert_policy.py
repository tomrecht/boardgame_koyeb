"""Tests for revert_policy.RevertPolicy (python3 test_revert_policy.py)."""
import json, os, sys, tempfile
from revert_policy import RevertPolicy


def R(outcome, diff=0.0, pm=0.0, games=200, keep=False, confirm=None, promote=False):
    return {'outcome': outcome, 'diff': diff, 'prescreen_mean': pm, 'prescreen_games': games,
            'keep': keep, 'promote': promote, 'confirm': confirm}


def main():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f'  {name}: {got} (want {want}) {"ok" if good else "FAIL"}')

    print('strict: the gate decides')
    p = RevertPolicy('strict')
    check('strict cap +', p.apply(R('cap', 0.1, keep=True), 50)['keep'], True)
    check('strict prescreen -0.01', p.apply(R('prescreen_reject', pm=-0.01), 50)['keep'], False)

    print('lenient burn-in')
    p = RevertPolicy('lenient', burnin=10, streak=3)
    check('burn-in panel reject kept', p.apply(R('reject', -0.5), 3)['keep'], True)
    check('burn-in early prescreen reject kept', p.apply(R('prescreen_reject', pm=-0.6, games=80), 4)['keep'], True)
    check('burn-in collapse reset', p.apply(R('prescreen_reject', pm=-5.0, games=40), 5)['keep'], False)

    print('lenient after burn-in')
    p = RevertPolicy('lenient', burnin=10, streak=3)
    check('full prescreen -0.01 kept', p.apply(R('prescreen_reject', pm=-0.01), 20)['keep'], True)
    check('early prescreen reject reset', p.apply(R('prescreen_reject', pm=-0.5, games=80), 21)['keep'], False)
    check('panel reject reset', p.apply(R('reject', -0.4), 22)['keep'], False)
    check('cap -0.1 kept (streak 1)', p.apply(R('cap', -0.1), 23)['keep'], True)
    check('cap -0.1 kept (streak 2)', p.apply(R('cap', -0.1), 24)['keep'], True)
    check('cap -0.1 reset (streak 3)', p.apply(R('cap', -0.1), 25)['keep'], False)
    check('streak restarted after reset', p.behind, 0)
    check('cap -0.1 kept again', p.apply(R('cap', -0.1), 26)['keep'], True)
    check('level resets streak', (p.apply(R('cap', 0.05), 27)['keep'], p.behind), (True, 0))
    check('confirm fail counts behind', (p.apply(R('confirm_fail', 0.3, confirm={'passed': False}), 28)['keep'], p.behind), (True, 1))
    check('promotion untouched, streak 0', (p.apply(R('promote', 0.3, keep=True, promote=True), 29)['keep'], p.behind), (True, 0))

    print('resume keeps the streak')
    d = tempfile.mkdtemp()
    sp = os.path.join(d, 's.json')
    p = RevertPolicy('lenient', burnin=0, streak=3, state_path=sp)
    p.apply(R('cap', -0.1), 5); p.apply(R('cap', -0.1), 6)
    p2 = RevertPolicy('lenient', burnin=0, streak=3, state_path=sp)
    check('resumed streak', p2.behind, 2)
    check('third in a row resets after resume', p2.apply(R('cap', -0.1), 7)['keep'], False)

    print('\nALL OK' if ok else '\nFAILURES')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
