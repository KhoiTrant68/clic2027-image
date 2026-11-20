"""Phase 0 summary: per quality, CLIC30 estimated bpp of every variant against MLIC++ (R) and GC-current (V0).

    python analyze.py --work GCE_DIR [--out DIR]

gap closed = (bpp(V0) - bpp(V)) / (bpp(V0) - bpp(R)): share of the GC -> MLIC++ entropy gap a variant recovers.
Gate G0 of the design: if R is < 3 % below V0 the entropy model is not where MLIC++'s advantage comes from; otherwise
pick the cheapest variant that closes >= 70 % of the gap with a forward time <= 2x V0.
"""
import argparse
import json
from pathlib import Path

ORDER = ['V0', 'V0f', 'V1', 'V2', 'V3', 'V3s']
DESC = {'V0': 'GC hiện tại (số nguyên)', 'V0f': 'GC hiện tại, float', 'V1': '+ E1 5×5 cục bộ + E2 mạng sâu',
        'V2': '+ E3 toàn cục giữa slice', 'V3': '+ E4 toàn cục trong slice', 'V3s': 'V3 với 10 slice × 32'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    out = a.out or a.work
    summary, md = {}, ['# Pha 0: entropy model trên latent MLIC++ cố định (CLIC30, bpp ước lượng)', '']
    for qd in sorted(a.work.glob('q*')):
        res = {v: json.loads((qd / v / 'result.json').read_text()) for v in ORDER if (qd / v / 'result.json').exists()}
        if not res:
            continue
        R = next(iter(res.values()))['mlicpp_bpp']
        v0 = res.get('V0')
        rows = []
        for v, r in res.items():
            row = dict(variant=v, bpp=r['bpp'], vs_R_pct=100 * (r['bpp'] / R - 1), params=r['params'],
                       forward_s=r['forward_s'], steps=r['steps'])
            if v0:
                row['vs_V0_pct'] = 100 * (r['bpp'] / v0['bpp'] - 1)
                row['forward_x'] = r['forward_s'] / v0['forward_s']
                gap = v0['bpp'] - R
                row['gap_closed_pct'] = 100 * (v0['bpp'] - r['bpp']) / gap if gap > 0 else None
            rows.append(row)
        gate = None
        if v0:
            g = 100 * (1 - R / v0['bpp'])
            ok = [r for r in rows if r['variant'] not in ('V0', 'V0f') and (r.get('gap_closed_pct') or 0) >= 70
                  and r['forward_x'] <= 2]
            ok.sort(key=lambda r: (r['forward_x'], -r['gap_closed_pct']))
            gate = dict(R_below_V0_pct=g, entropy_is_source=g >= 3, pick=ok[0]['variant'] if ok else None)
        if gate:  # design 2.1: at 50k steps the gate is reported but only selects; G0 is decided on the long runs
            gate['note'] = '50k steps: selection only (design 2.1); G0 is decided on the 300k-step runs below'
        summary[qd.name] = dict(R_bpp=R, rows=rows, gate=gate)
        long_md = []
        pick = (qd / 'long_pick.txt').read_text().strip() if (qd / 'long_pick.txt').exists() else None
        lv0, lp = qd / 'long_V0' / 'result.json', qd / f'long_{pick}' / 'result.json'
        if pick and lv0.exists() and lp.exists():
            b0, bp = json.loads(lv0.read_text()), json.loads(lp.read_text())
            g = 100 * (1 - R / b0['bpp'])
            gap = b0['bpp'] - R
            closed = 100 * (b0['bpp'] - bp['bpp']) / gap if gap > 0 else None
            fx = bp['forward_s'] / b0['forward_s']
            if g < 3:
                verdict = 'G0: dừng (R thấp hơn V0 dưới 3%, entropy không phải nguồn lợi thế)'
            elif closed is not None and closed >= 70 and fx <= 2:
                verdict = f'G0: đạt, chọn {pick}'
            else:
                verdict = f'G0: không đạt ({pick} lấy lại {closed:.0f}% khoảng cách, forward {fx:.2f}× V0)'
            summary[qd.name]['long'] = dict(pick=pick, V0=b0['bpp'], pick_bpp=bp['bpp'], steps=b0['steps'],
                                            R_below_V0_pct=g, gap_closed_pct=closed, forward_x=fx, verdict=verdict)
            long_md = [f"**300k bước (sửa đổi 2.1):** V0 {b0['bpp']:.4f}, {pick} {bp['bpp']:.4f} bpp; R thấp hơn V0 "
                   f"{g:.2f}%; {pick} lấy lại {closed:.0f}% khoảng cách; forward {fx:.2f}× V0. **{verdict}**", '']
        md += [f'## {qd.name} (MLIC++ R = {R:.4f} bpp)', '',
               '| Biến thể | Mô tả | bpp | so với R | so với V0 | lấy lại khoảng cách | forward ×V0 | tham số |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            gc = r.get('gap_closed_pct')
            md.append(f"| {r['variant']} | {DESC[r['variant']]} | {r['bpp']:.4f} | {r['vs_R_pct']:+.2f}% | "
                      f"{r.get('vs_V0_pct', 0):+.2f}% | {'' if gc is None else f'{gc:.0f}%'} | "
                      f"{r.get('forward_x', float('nan')):.2f} | {r['params'] / 1e6:.1f} M |")
        if gate:
            md += ['', f"(50k bước, chỉ để chọn biến thể) R thấp hơn V0 {gate['R_below_V0_pct']:.2f}%; tiêu chí G0 áp ở 50k: "
                       f"{'entropy là nguồn lợi thế' if gate['entropy_is_source'] else 'entropy KHÔNG phải nguồn lợi thế (< 3%)'}; "
                       f"biến thể chọn: {gate['pick'] or 'không có biến thể đạt ≥ 70% với forward ≤ 2× V0'}", '']
        md += long_md
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
