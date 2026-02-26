#!/usr/bin/env python3
"""
주식 매수/매도 시뮬레이터
- 벤포드 법칙 + 기술적 지표 기반 매수 시그널
- Walk-forward 백테스트 (+21% 익절 / -7% 손절)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators
from modules.backtester import run_backtest, summarize_trades


def print_header(title, char='='):
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_trade(i, t):
    """개별 거래 출력"""
    result_mark = {'WIN': '✅ 승', 'LOSS': '❌ 패', 'OPEN': '⏳ 진행중'}
    mark = result_mark.get(t['result'], t['result'])

    entry = t['entry_date'].strftime('%Y-%m-%d')
    exit_d = t['exit_date'].strftime('%Y-%m-%d')

    print(f"  #{i+1:3d} | {mark} | 매수: {entry} @ {t['entry_price']:>8,}원"
          f" → 매도: {exit_d} @ {t['exit_price']:>8,}원"
          f" | 수익: {t['return_pct']:>+7.2f}% | {t['holding_days']:>3d}일"
          f" | 스코어: {t['score']:.1f}")

    if t['details']:
        detail_str = ', '.join(f"{v}" for v in t['details'].values())
        print(f"        근거: {detail_str}")


def run(xml_path, buy_threshold=4.0, take_profit=0.21, stop_loss=0.07,
        cooldown=5, benford_window=30):
    """메인 실행 함수"""

    # 1. 데이터 로드
    print_header("데이터 로드")
    df, symbol, name = parse_stock_xml(xml_path)
    print(f"  종목: {name} ({symbol})")
    print(f"  기간: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  총 거래일: {len(df):,}일")
    print(f"  가격 범위: {df['low'].min():,}원 ~ {df['high'].max():,}원")

    # 2. 기술지표 계산
    print_header("기술지표 계산")
    df = calc_all_indicators(df)
    print("  이동평균(5/20/60), RSI(14), 볼린저밴드(20,2σ)")
    print("  MACD(12/26/9), 거래량비율(20), 캔들스틱패턴")
    print("  벤포드 법칙 (거래량/가격변동 이상탐지)")

    # 3. 백테스트 실행
    print_header("백테스트 설정")
    print(f"  매수 임계값  : 스코어 >= {buy_threshold}")
    print(f"  익절 기준    : +{take_profit*100:.0f}%")
    print(f"  손절 기준    : -{stop_loss*100:.0f}%")
    print(f"  쿨다운       : {cooldown}일")
    print(f"  벤포드 윈도우: {benford_window}일")

    print_header("백테스트 실행중...")
    trades = run_backtest(df, buy_threshold, take_profit, stop_loss,
                          cooldown, benford_window)

    # 4. 결과 요약
    summary = summarize_trades(trades)

    print_header("📊 백테스트 결과", '━')

    if summary['total'] == 0:
        print("  매수 시그널이 발생하지 않았습니다.")
        print(f"  임계값({buy_threshold})을 낮춰보세요.")
        return trades, summary

    print(f"""
  총 시그널      : {summary['total']}건
  완료 거래      : {summary['closed']}건 (승: {summary['wins']} / 패: {summary['losses']})
  미완료 거래    : {summary['open']}건

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ★ 승률         : {summary['win_rate']:.1f}%
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  평균 수익률    : {summary['avg_return']:+.2f}%
  누적 수익률    : {summary['total_return_pct']:+.2f}%
  평균 보유기간  : {summary['avg_holding']:.0f}일""")

    if summary['best_trade']:
        bt = summary['best_trade']
        print(f"\n  최고 거래: {bt['entry_date'].strftime('%Y-%m-%d')} "
              f"매수 {bt['entry_price']:,}원 → {bt['return_pct']:+.2f}%")
    if summary['worst_trade']:
        wt = summary['worst_trade']
        print(f"  최악 거래: {wt['entry_date'].strftime('%Y-%m-%d')} "
              f"매수 {wt['entry_price']:,}원 → {wt['return_pct']:+.2f}%")

    # 5. 개별 거래 상세
    print_header("📋 거래 상세 내역", '─')
    for i, t in enumerate(trades):
        print_trade(i, t)
        if i < len(trades) - 1:
            print("  " + "·" * 66)

    return trades, summary


def run_optimization(xml_path, take_profit=0.21, stop_loss=0.07):
    """다양한 임계값으로 자동 최적화"""
    print_header("자동 임계값 탐색", '◆')

    df, symbol, name = parse_stock_xml(xml_path)
    df = calc_all_indicators(df)

    thresholds = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    cooldowns = [3, 5, 7, 10]

    print(f"\n  {'임계값':>6} | {'쿨다운':>4} | {'거래수':>4} | {'승':>3} | {'패':>3} |"
          f" {'승률':>6} | {'누적수익':>8} | {'평균보유':>6}")
    print("  " + "─" * 65)

    best = {'win_rate': 0, 'threshold': 0, 'cooldown': 0, 'trades': 0}

    for th in thresholds:
        for cd in cooldowns:
            trades = run_backtest(df, th, take_profit, stop_loss, cd)
            s = summarize_trades(trades)
            if s['closed'] > 0:
                marker = ' ◀' if s['win_rate'] >= 80 and s['closed'] >= 3 else ''
                print(f"  {th:>6.1f} | {cd:>4} | {s['closed']:>4} | "
                      f"{s['wins']:>3} | {s['losses']:>3} | "
                      f"{s['win_rate']:>5.1f}% | {s['total_return_pct']:>+7.1f}% | "
                      f"{s['avg_holding']:>5.0f}일{marker}")

                if (s['win_rate'] > best['win_rate'] and s['closed'] >= 3) or \
                   (s['win_rate'] == best['win_rate'] and s['closed'] > best['trades']):
                    best = {'win_rate': s['win_rate'], 'threshold': th,
                            'cooldown': cd, 'trades': s['closed']}

    print("  " + "─" * 65)
    if best['trades'] > 0:
        print(f"\n  최적 조합: 임계값={best['threshold']}, 쿨다운={best['cooldown']}일"
              f" → 승률 {best['win_rate']:.1f}% ({best['trades']}건)")
    return best


if __name__ == '__main__':
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'kakaopay_stock.xml')

    if not os.path.exists(xml_file):
        print(f"오류: 파일을 찾을 수 없습니다 → {xml_file}")
        sys.exit(1)

    # 1단계: 최적 임계값 탐색
    best = run_optimization(xml_file)

    # 2단계: 최적 파라미터로 상세 분석
    if best['trades'] > 0:
        trades, summary = run(xml_file,
                              buy_threshold=best['threshold'],
                              cooldown=best['cooldown'])
    else:
        trades, summary = run(xml_file)

    wr = summary.get('win_rate', 0)
    print_header("분석 코멘트", '─')
    if wr >= 80:
        print(f"  목표 승률 80% 달성! ({wr:.1f}%)")
        print(f"  단, 거래 횟수({summary['closed']}건)가 적으면 통계적 신뢰도가 낮습니다.")
    elif wr >= 60:
        print(f"  승률 {wr:.1f}% — 임계값을 올리면 승률 상승 가능 (거래수 감소).")
    elif wr >= 40:
        print(f"  승률 {wr:.1f}% — 규칙 조합 재검토가 필요합니다.")
    else:
        print(f"  승률 {wr:.1f}% — +21%/-7% 비대칭 하에서 이 종목은 어려운 구간입니다.")
        print("  장기 하락 종목에서 +21% 달성은 구조적으로 불리합니다.")
