#!/usr/bin/env python3
"""
전략 비교 검증: 기존 default vs 세력 추종형 (force_following)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators
from modules.backtester import run_backtest, summarize_trades

XML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xml')

STRATEGIES = {
    'default(기존)': dict(
        buy_threshold=4.0, take_profit=0.17, stop_loss=0.07,
        cooldown=3, benford_window=30, profile_name='default'
    ),
    'force_following(세력)': dict(
        buy_threshold=4.0, take_profit=0.21, stop_loss=0.07,
        cooldown=5, benford_window=30, profile_name='force_following'
    ),
}

def calc_profit(trades):
    """투자금 기반 총 수익금 계산 (1종목 10주 기준)"""
    total = 0
    for t in trades:
        shares = 10
        profit = (t['exit_price'] - t['entry_price']) * shares
        total += profit
    return total

def run_all(strategy_params):
    results = []
    xml_files = sorted([f for f in os.listdir(XML_DIR)
                        if f.endswith('.xml') and f != 'KOSPI.xml'])
    total = len(xml_files)
    for i, fname in enumerate(xml_files):
        path = os.path.join(XML_DIR, fname)
        try:
            df, symbol, name = parse_stock_xml(path)
            df = calc_all_indicators(df)
            trades = run_backtest(df, **strategy_params)
            if not trades:
                continue
            s = summarize_trades(trades)
            closed = s.get('closed', 0)
            if closed < 3:
                continue
            results.append({
                'symbol': symbol, 'name': name,
                'total': s['total'],
                'wins': s['wins'],
                'losses': s['losses'],
                'opens': s.get('open', 0),
                'closed': closed,
                'win_rate': s['win_rate'],
                'avg_return': s['avg_return'],
                'total_profit': calc_profit(trades),
                'cum_return': s['total_return_pct'],
            })
        except Exception:
            pass
        if (i+1) % 20 == 0:
            print(f"  진행: {i+1}/{total}...")
    return results

def print_summary(name, results):
    if not results:
        print(f"  {name}: 결과 없음")
        return
    total_closed = sum(r['closed'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    avg_wr = total_wins / total_closed * 100 if total_closed else 0
    avg_ret = sum(r['avg_return'] for r in results) / len(results)
    total_profit = sum(r['total_profit'] for r in results)
    stocks_60 = sum(1 for r in results if r['win_rate'] >= 60)
    stocks_70 = sum(1 for r in results if r['win_rate'] >= 70)
    avg_signals = sum(r['total'] for r in results) / len(results)

    print(f"\n{'='*58}")
    print(f"  전략: {name}")
    print(f"{'='*58}")
    print(f"  분석 종목 수     : {len(results)}개")
    print(f"  종목당 평균 신호 : {avg_signals:.1f}건")
    print(f"  총 완료 거래     : {total_closed:,}건")
    print(f"  전체 승률        : {avg_wr:.1f}%  ({total_wins:,}승 / {total_closed-total_wins:,}패)")
    print(f"  평균 수익률/건   : {avg_ret:+.2f}%")
    print(f"  총 수익금 합계   : {total_profit/10000:+,.0f}만원 (10주 기준)")
    print(f"  승률 60%↑ 종목   : {stocks_60}개 ({stocks_60/len(results)*100:.0f}%)")
    print(f"  승률 70%↑ 종목   : {stocks_70}개 ({stocks_70/len(results)*100:.0f}%)")

    top = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:10]
    print(f"\n  ▶ 상위 10 종목 (승률 기준)")
    print(f"  {'종목':<12} {'승률':>6} {'거래':>5} {'평균수익':>8} {'누적복리':>9}")
    print(f"  {'-'*50}")
    for r in top:
        print(f"  {r['name'][:10]:<12} {r['win_rate']:>5.1f}% {r['closed']:>5}건"
              f" {r['avg_return']:>+7.2f}% {r['cum_return']:>+8.1f}%")

if __name__ == '__main__':
    all_results = {}
    for strat_name, params in STRATEGIES.items():
        print(f"\n[{strat_name}] 백테스트 시작...")
        results = run_all(params)
        all_results[strat_name] = results
        print_summary(strat_name, results)

    print(f"\n\n{'='*58}")
    print(f"  📊 전략 비교 요약")
    print(f"{'='*58}")
    print(f"  {'전략':<28} {'승률':>6} {'평균수익':>8} {'60%↑종목':>8} {'총수익금':>12}")
    print(f"  {'-'*65}")
    for strat_name, results in all_results.items():
        if not results:
            continue
        total_closed = sum(r['closed'] for r in results)
        total_wins = sum(r['wins'] for r in results)
        avg_wr = total_wins / total_closed * 100 if total_closed else 0
        avg_ret = sum(r['avg_return'] for r in results) / len(results)
        total_profit = sum(r['total_profit'] for r in results)
        stocks_60 = sum(1 for r in results if r['win_rate'] >= 60)
        print(f"  {strat_name:<28} {avg_wr:>5.1f}% {avg_ret:>+7.2f}%"
              f" {stocks_60:>7}개 {total_profit/10000:>+10,.0f}만원")
    print()
