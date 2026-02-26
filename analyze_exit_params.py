#!/usr/bin/env python3
"""
다양한 익절/손절 비율에서 최적 승률을 탐색
현재 전략(정배열 + 고점근접 + MA20상승 + 양봉)을 기반으로
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators
from modules.backtester import run_backtest, summarize_trades


def main():
    xml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'kakaopay_stock.xml')
    df, symbol, name = parse_stock_xml(xml_file)
    df = calc_all_indicators(df)

    print(f"=== {name} ({symbol}) 익절/손절 비율별 승률 분석 ===\n")
    print(f"전략: 정배열 + 20일고점근접 + MA20상승 + 양봉 + 과열차단")
    print(f"쿨다운: 3일\n")

    # 다양한 익절/손절 조합 테스트
    take_profits = [0.07, 0.10, 0.12, 0.15, 0.18, 0.21, 0.25, 0.30]
    stop_losses = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]

    print(f"{'':>12}", end='')
    for sl in stop_losses:
        print(f" | 손절-{sl*100:.0f}%", end='')
    print()
    print("─" * (12 + len(stop_losses) * 10))

    best = {'wr': 0, 'tp': 0, 'sl': 0, 'trades': 0, 'ret': 0}

    for tp in take_profits:
        print(f"  익절+{tp*100:<4.0f}%", end='')
        for sl in stop_losses:
            trades = run_backtest(df, buy_threshold=3.0, take_profit=tp,
                                 stop_loss=sl, cooldown=3)
            s = summarize_trades(trades)

            if s['closed'] >= 3:
                wr = s['win_rate']
                marker = "★" if wr >= 80 else ("◆" if wr >= 70 else "")
                print(f" | {wr:4.0f}%/{s['closed']:2d}{marker}", end='')

                score = wr * 0.7 + min(s['closed'] / 20, 1) * 30  # 승률 + 거래수 밸런스
                if score > best.get('score', 0):
                    best = {'wr': wr, 'tp': tp, 'sl': sl,
                            'trades': s['closed'], 'ret': s['total_return_pct'],
                            'score': score}
            else:
                print(f" |   {'N/A':>5} ", end='')
        print()

    print("─" * (12 + len(stop_losses) * 10))
    print("  (형식: 승률%/거래수, ★=80%+, ◆=70%+)\n")

    # 80% 이상 달성한 조합 출력
    print("=" * 65)
    print("  80% 이상 승률 조합 (거래 3건 이상)")
    print("=" * 65)

    found_80 = False
    for tp in take_profits:
        for sl in stop_losses:
            trades = run_backtest(df, buy_threshold=3.0, take_profit=tp,
                                 stop_loss=sl, cooldown=3)
            s = summarize_trades(trades)
            if s['closed'] >= 3 and s['win_rate'] >= 80:
                found_80 = True
                print(f"  +{tp*100:.0f}%/-{sl*100:.0f}%  →  승률 {s['win_rate']:.1f}% "
                      f"| {s['closed']}건 (승:{s['wins']} 패:{s['losses']}) "
                      f"| 누적수익: {s['total_return_pct']:+.1f}%")

    if not found_80:
        print("  80% 이상 달성 조합 없음")

    print(f"\n  최적 밸런스: +{best['tp']*100:.0f}%/-{best['sl']*100:.0f}% → "
          f"승률 {best['wr']:.1f}%, {best['trades']}건, 누적 {best['ret']:+.1f}%")


if __name__ == '__main__':
    main()
