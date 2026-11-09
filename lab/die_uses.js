/* What each die value does for the side to move: node lab/die_uses.js "<pieces>" [layout=sum7]
   pieces: "w:x@G1,2@G2,5@4.8;b:x@3.11" (G<n> = goal numbered n on that layout;
   the rest of each side is saved). Lists, per die 1-6 used ALONE: saves, goal
   arrivals, captures, walls built. */
const { staticFor, engineFrom } = require('./eg_common.js');
const [spec, layout = 'sum7'] = process.argv.slice(2);
const sd = staticFor(layout === 'original' ? null : layout);
const idx = (loc) => {
    if (loc[0] === 'G') return sd.tile_types.findIndex((t, i) => t === 'save' && sd.tile_numbers[i] === +loc.slice(1));
    const [r, s] = loc.split('.').map(Number);
    return sd.tile_coords.findIndex(c => c[0] === r && c[1] === s);
};
const st = { racks: { white_unentered: [], black_unentered: [], white_saved: [], black_saved: [] },
             occupancy: {}, dice: [[1, false], [1, false]], current_player: 'white',
             stages: { white: 'midgame', black: 'midgame' }, first_move: null, draw_called: false };
for (const part of spec.split(';')) {
    const [side, list] = part.split(':');
    const pl = side === 'w' ? 'white' : 'black';
    const used = new Set();
    let blank = 7;
    for (const item of list.split(',').filter(Boolean)) {
        const [n, loc] = item.split('@');
        const num = n === 'x' ? blank++ : +n;
        used.add(num);
        const i = idx(loc);
        if (i < 0) throw new Error('no tile ' + loc);
        (st.occupancy[i] = st.occupancy[i] || []).push([pl, num]);
    }
    for (let k = 1; k <= 12; k++) if (!used.has(k)) st.racks[pl + '_saved'].push(k);
}
const where = (e, t) => { const [r, s] = sd.tile_coords[t]; return sd.tile_types[t] === 'save' ? 'G' + sd.tile_numbers[t] : r + '.' + s; };
for (let d = 1; d <= 6; d++) {
    st.dice = [[d, false], [d === 6 ? 5 : 6, true]];       // one live die
    const e = engineFrom(sd, st);
    const uses = [];
    for (const m of e.getValidMoves()) {
        if (m.piece === null || m.dest === 0) continue;
        const p = e.find(m.piece[0], m.piece[1]);
        const from = p.tile >= 0 ? where(e, p.tile) : 'rack';
        const label = (p.number > 6 ? 'x' : p.number) + '@' + from;
        if (m.dest === 'save') { uses.push('save ' + label); continue; }
        const t = sd.tile_coords.findIndex(c => c[0] === m.dest[0] && c[1] === m.dest[1]);
        const occ = e.occ[t];
        if (sd.tile_types[t] === 'save') uses.push(label + ' to ' + where(e, t));
        else if (occ.length === 1 && occ[0].player !== 'white') uses.push(label + ' captures on ' + where(e, t));
        else if (occ.length >= 1 && occ[0].player === 'white') uses.push(label + ' walls ' + where(e, t));
    }
    console.log(d + ': ' + (uses.join(' | ') || '-'));
}
