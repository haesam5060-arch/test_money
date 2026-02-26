from modules.signal_engine import calculate_buy_score
from modules.regime_filter import load_kospi, is_bear_market

# KOSPI 데이터 로드 (xml/KOSPI.xml 자동 탐색)
load_kospi()


def run_backtest(df, buy_threshold=4.0, take_profit=0.17, stop_loss=0.07,
                 cooldown=5, benford_window=30, profile_name='default',
                 use_regime_filter=True):
    """
    Walk-forward 백테스트 실행

    매일봉을 과거→현재 순서로 순회하며:
    1. KOSPI 시장 국면 확인 — 약세 국면이면 매수 차단 (Option A)
    2. 해당 일자까지의 데이터만 사용하여 매수 스코어 계산
    3. 스코어 >= 임계값이면 종가 매수
    4. 이후 일봉에서 +take_profit% 또는 -stop_loss% 도달 시 청산
    5. 같은 날 둘 다 도달 시 시가 기준으로 판단

    Parameters:
        df: 주가 DataFrame (indicators 계산 완료 상태)
        buy_threshold: 매수 시그널 발생 임계값
        take_profit: 익절 비율 (0.17 = +17%)
        stop_loss: 손절 비율 (0.07 = -7%)
        cooldown: 매수 후 다음 매수까지 대기 일수
        benford_window: 벤포드 분석 윈도우
        profile_name: 종목 유형 프로필 ('default' 또는 'large_cap')
        use_regime_filter: True면 KOSPI 약세 국면에서 매수 차단 (Option A)
    """
    trades = []
    last_buy_idx = -cooldown

    for idx in range(60, len(df)):
        # 쿨다운 체크
        if idx - last_buy_idx < cooldown:
            continue

        score, details = calculate_buy_score(df, idx, benford_window, profile_name)

        if score < buy_threshold:
            continue

        entry_date = df.iloc[idx]['date']

        # Option A: KOSPI 약세 국면 Hard Block
        if use_regime_filter and is_bear_market(entry_date):
            continue

        entry_price = df.iloc[idx]['close']
        target_price = entry_price * (1 + take_profit)
        stop_price = entry_price * (1 - stop_loss)

        result = None
        exit_date = None
        exit_price = None

        # 매수 다음날부터 청산 조건 탐색
        for j in range(idx + 1, len(df)):
            day = df.iloc[j]

            hit_stop = day['low'] <= stop_price
            hit_target = day['high'] >= target_price

            if hit_target and hit_stop:
                # 같은 날 두 가격 모두 도달 → 시가 기준 판단
                if day['open'] <= stop_price:
                    result = 'LOSS'
                    exit_price = stop_price
                elif day['open'] >= target_price:
                    result = 'WIN'
                    exit_price = target_price
                else:
                    # 보수적 판단: 손절 우선
                    result = 'LOSS'
                    exit_price = stop_price
                exit_date = day['date']
                break
            elif hit_target:
                result = 'WIN'
                exit_price = target_price
                exit_date = day['date']
                break
            elif hit_stop:
                result = 'LOSS'
                exit_price = stop_price
                exit_date = day['date']
                break

        if result is None:
            # 데이터 끝까지 청산 조건 미도달 → OPEN
            result = 'OPEN'
            exit_date = df.iloc[-1]['date']
            exit_price = df.iloc[-1]['close']

        holding_days = (exit_date - entry_date).days if exit_date else 0
        return_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_date': entry_date,
            'entry_price': int(entry_price),
            'exit_date': exit_date,
            'exit_price': int(exit_price),
            'target_price': int(target_price),
            'stop_price': int(stop_price),
            'score': round(score, 2),
            'details': details,
            'result': result,
            'return_pct': round(return_pct, 2),
            'holding_days': holding_days,
        })

        last_buy_idx = idx

    return trades


def summarize_trades(trades):
    """거래 결과 요약 통계"""
    if not trades:
        return {
            'total': 0, 'wins': 0, 'losses': 0, 'open': 0,
            'win_rate': 0.0, 'avg_return': 0.0, 'avg_holding': 0,
        }

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    opens = [t for t in trades if t['result'] == 'OPEN']
    closed = wins + losses

    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_return = sum(t['return_pct'] for t in closed) / len(closed) if closed else 0
    avg_holding = sum(t['holding_days'] for t in closed) / len(closed) if closed else 0

    # 수익률 계산 (복리)
    total_return = 1.0
    for t in sorted(closed, key=lambda x: x['entry_date']):
        total_return *= (1 + t['return_pct'] / 100)
    total_return_pct = (total_return - 1) * 100

    return {
        'total': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'open': len(opens),
        'closed': len(closed),
        'win_rate': round(win_rate, 2),
        'avg_return': round(avg_return, 2),
        'avg_holding': round(avg_holding, 1),
        'total_return_pct': round(total_return_pct, 2),
        'best_trade': max(closed, key=lambda x: x['return_pct']) if closed else None,
        'worst_trade': min(closed, key=lambda x: x['return_pct']) if closed else None,
    }
