#!/usr/bin/env python3
"""
대규모 종목 분석 최적화 시스템
- 200+ 한국 주요 종목 XML 자동 다운로드
- index.html JS 로직 1:1 Python 포팅
- 파라미터 + 스코어링 가중치 최적화
- 기존 파일 수정 없음 (독립 실행)
"""
import os
import re
import ssl
import sys
import json
import math
import time
import copy
import random
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import defaultdict

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ============================================================
# 0. 상수 & 종목 코드
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(BASE_DIR, 'xml')
NAVER_API = 'https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count=3000&requestType=0'

STOCK_CODES = {
    # KOSPI 대형주
    '005930': '삼성전자', '000660': 'SK하이닉스', '005380': '현대차',
    '005490': 'POSCO홀딩스', '105560': 'KB금융', '207940': '삼성바이오로직스',
    '373220': 'LG에너지솔루션', '006400': '삼성SDI', '068270': '셀트리온',
    '035420': '네이버', '035720': '카카오', '055550': '신한지주',
    '003550': 'LG', '066570': 'LG전자', '051910': 'LG화학',
    '034730': 'SK', '000270': '기아', '032830': '삼성생명',
    '096770': 'SK이노베이션', '003670': '포스코퓨처엠', '010950': 'S-Oil',
    '009150': '삼성전기', '086790': '하나금융지주', '015760': '한국전력',
    '018260': '삼성에스디에스', '033780': 'KT&G', '012330': '현대모비스',
    '316140': '우리금융지주', '329180': 'HD현대중공업', '009540': 'HD한국조선해양',
    '010130': '고려아연', '028260': '삼성물산', '034020': '두산에너빌리티',
    '030200': 'KT', '017670': 'SK텔레콤', '000810': '삼성화재',
    '036570': '엔씨소프트', '251270': '넷마블', '259960': '크래프톤',
    '352820': '하이브', '012450': '한화에어로스페이스',
    '000720': '현대건설', '003490': '대한항공', '042660': '한화오션',
    '011200': 'HMM', '011170': '롯데케미칼', '006360': 'GS건설',
    '004020': '현대제철', '005830': 'DB손해보험', '024110': '기업은행',
    '138930': 'BNK금융지주', '139480': '이마트', '004170': '신세계',
    '090430': '아모레퍼시픽', '097950': 'CJ제일제당', '271560': '오리온',
    '000100': '유한양행', '128940': '한미약품', '006800': '미래에셋증권',
    '005940': 'NH투자증권', '016360': '삼성증권', '039490': '키움증권',
    '161390': '한국타이어앤테크놀로지', '011790': 'SKC',
    '036460': '한국가스공사', '078930': 'GS', '021240': '코웨이',
    '008770': '호텔신라', '002790': '아모레G', '326030': 'SK바이오팜',
    # KOSPI 중형주
    '377300': '카카오페이', '323410': '카카오뱅크',
    '005935': '삼성전자우', '009830': '한화솔루션', '047050': '포스코인터내셔널',
    '010140': '삼성중공업', '267250': 'HD현대', '042670': '두산인프라코어',
    '008930': '한미사이언스', '007070': 'GS리테일', '069500': 'KODEX200',
    '030000': '제일기획', '035250': '강원랜드', '051900': 'LG생활건강',
    '064350': '현대로템', '111770': '영원무역', '180640': '한진칼',
    '241560': '두산밥캣', '005850': 'DB하이텍', '092780': '동양생명',
    # KOSDAQ 대형주
    '247540': '에코프로비엠', '086520': '에코프로', '196170': '알테오젠',
    '028300': 'HLB', '041510': '에스엠', '293490': '카카오게임즈',
    '263750': '펄어비스', '112040': '위메이드', '078340': '컴투스',
    '042700': '한미반도체', '058470': '리노공업', '035900': 'JYP엔터',
    '068760': '셀트리온헬스케어', '091990': '셀트리온제약',
    '145020': '휴젤', '214150': '클래시스', '357780': '솔브레인',
    '036930': '주성엔지니어링', '950160': '코오롱티슈진',
    '383220': 'F&F', '033640': '네패스', '048410': '현대바이오',
    '039030': '이오테크닉스', '222160': 'NPX반도체',
    '095340': 'ISC', '240810': '원익IPS', '131970': '테스나',
    '060310': '3S', '441270': '파두', '067160': '아프리카TV',
    '005290': '동진쎄미켐', '140860': '파크시스템스',
    '257720': '실리콘투', '367340': '시공테크',
    # KOSDAQ 중형주
    '099190': '아이센스', '090460': '비에이치', '352480': '씨이랩',
    '336260': '두산퓨얼셀', '314930': '바이오노트', '337840': '유진로봇',
    '069080': '웹젠', '192820': '코스맥스', '194480': '데브시스터즈',
    '141080': '레고켐바이오', '226330': '신테카바이오',
    '053800': '안랩', '060280': '큐렉소', '238090': '앤디포스',
    '048260': '오스템임플란트', '215600': '신라젠',
    '122870': '와이지엔터테인먼트', '101000': '상상인인더스트리',
    '054950': '제이브이엠', '237690': 'SG세계물산',
    '108230': '톱텍', '950170': '코오롱생명과학',
    '174900': '앱클론', '365340': '성일하이텍', '322000': '현대에너지솔루션',
    '048870': '시너지이노베이션', '222800': '심텍',
    '298380': '에이비엘바이오', '900140': '코라오홀딩스',
    # 추가 중소형주
    '002380': 'KCC', '001040': 'CJ', '023530': '롯데쇼핑',
    '006650': '대한유화', '007310': '오뚜기', '014680': '한솔케미칼',
    '016380': 'KG동부제철', '020150': '일진머티리얼즈', '025540': '한국단자',
    '029780': '삼성카드', '032640': 'LG유플러스', '034220': 'LG디스플레이',
    '035760': 'CJ ENM', '036090': '위지트', '044090': '텔레칩스',
    '047810': '한국항공우주', '051600': '한전KPS', '052690': '한전기술',
    '055490': '테이팩스', '056190': '에스에프에이', '064960': 'S&T모티브',
    '071050': '한국금융지주', '078000': '텔레필드', '088350': '한화생명',
    '100840': 'SNT에너지', '101530': '해태제과식품', '114090': 'GKL',
    '120110': '코오롱인더', '128820': '데상트', '130960': 'CJ CGV',
    '138040': '메리츠금융지주', '161890': '한국콜마',
    '185750': '종근당', '192080': '더블유게임즈', '214370': '케어젠',
    '278280': '천보', '285130': 'SK케미칼', '298000': '효성중공업',
    '298040': '효성첨단소재', '307950': '현대오토에버',
}

BENFORD = [0, 0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]

DEFAULT_WEIGHTS = {
    # Pattern A: 과매도 반등
    'rsi_extreme_oversold': 3.5,
    'rsi_oversold': 3.0,
    'rsi_mild_oversold': 2.0,
    'rsi_neutral': 1.0,
    'rsi_momentum': 0.5,
    'rsi_overbought_penalty': 1.0,
    'bb_below': 2.5,
    'bb_lower': 2.0,
    'bb_low': 1.0,
    'bb_high_penalty': 0.5,
    'dip_sharp': 2.5,
    'dip_moderate': 2.0,
    'dip_mild': 1.0,
    'reversal_strong': 2.0,
    'reversal_moderate': 1.5,
    'reversal_mild': 0.5,
    'low20_near': 1.5,
    'low20_vicinity': 0.8,
    'ma20_gap_large': 1.5,
    'ma20_gap_moderate': 1.0,
    # Pattern B: 추세 추종
    'align_full': 2.0,
    'align_partial': 0.8,
    'slope_steep': 1.5,
    'slope_moderate': 1.0,
    'slope_mild': 0.3,
    'high20_near': 1.0,
    # 공통
    'macd_golden_cross': 1.5,
    'vol_healthy': 1.0,
    'engulfing': 1.0,
    # 리스크
    'surge_risk': 1.5,
}


# ============================================================
# 1. 데이터 다운로드 & 파싱
# ============================================================
def download_stock_xml(code, force=False):
    filepath = os.path.join(XML_DIR, f'{code}.xml')
    if not force and os.path.exists(filepath):
        age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
        if age_hours < 24:
            return filepath
    url = NAVER_API.format(code=code)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            content = resp.read()
        with open(filepath, 'wb') as f:
            f.write(content)
        return filepath
    except Exception as e:
        return None


def download_all_stocks(stock_codes):
    os.makedirs(XML_DIR, exist_ok=True)
    results = {}
    total = len(stock_codes)
    for i, (code, name) in enumerate(stock_codes.items()):
        sys.stdout.write(f'\r  [{i+1}/{total}] {name} ({code})...')
        sys.stdout.flush()
        path = download_stock_xml(code)
        if path:
            results[code] = path
    print(f'\r  다운로드 완료: {len(results)}/{total} 종목' + ' ' * 30)
    return results


def parse_naver_xml(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    for enc in ['euc-kr', 'utf-8', 'cp949']:
        try:
            content = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        content = raw.decode('utf-8', errors='replace')
    content = re.sub(r'encoding="[^"]*"', 'encoding="UTF-8"', content)
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None
    chartdata = root.find('.//chartdata')
    if chartdata is None:
        for tag in root.iter():
            if 'chartdata' in tag.tag.lower():
                chartdata = tag
                break
    if chartdata is None:
        return None
    symbol = chartdata.get('symbol', '')
    name = chartdata.get('name', symbol)
    data = []
    for item in chartdata.findall('.//item'):
        parts = item.get('data', '').split('|')
        if len(parts) == 6:
            try:
                ds = parts[0]
                data.append({
                    'date': f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}',
                    'open': int(parts[1]), 'high': int(parts[2]),
                    'low': int(parts[3]), 'close': int(parts[4]),
                    'volume': int(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    if not data:
        return None
    data.sort(key=lambda d: d['date'])
    return {'symbol': symbol, 'name': name, 'data': data}


# ============================================================
# 2. 기술지표 계산 (JS calcIndicators 1:1 포팅)
# ============================================================
def calc_indicators(data):
    n = len(data)
    close = [d['close'] for d in data]

    # 이동평균
    for w in [5, 20, 60]:
        for i in range(n):
            if i < w - 1:
                data[i][f'ma{w}'] = None
            else:
                data[i][f'ma{w}'] = sum(close[i - w + 1:i + 1]) / w

    # RSI (Wilder)
    period = 14
    rsi = [None] * n
    if n > period:
        gains, losses = [], []
        for i in range(1, period + 1):
            d = close[i] - close[i - 1]
            gains.append(d if d > 0 else 0)
            losses.append(-d if d < 0 else 0)
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        rsi[period] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        for i in range(period + 1, n):
            d = close[i] - close[i - 1]
            avg_gain = (avg_gain * (period - 1) + (d if d > 0 else 0)) / period
            avg_loss = (avg_loss * (period - 1) + (-d if d < 0 else 0)) / period
            rsi[i] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(n):
        data[i]['rsi'] = rsi[i]

    # 볼린저 밴드 (20, 2σ) — population std
    for i in range(n):
        if i < 19:
            data[i]['bbMid'] = data[i]['bbUpper'] = data[i]['bbLower'] = None
        else:
            window = close[i - 19:i + 1]
            mid = sum(window) / 20
            sq = sum((x - mid) ** 2 for x in window)
            std = (sq / 20) ** 0.5
            data[i]['bbMid'] = mid
            data[i]['bbUpper'] = mid + 2 * std
            data[i]['bbLower'] = mid - 2 * std

    # 거래량 비율
    for i in range(n):
        if i < 19:
            data[i]['volRatio'] = data[i]['volAvg'] = None
        else:
            vol_sum = sum(data[j]['volume'] for j in range(i - 19, i + 1))
            data[i]['volAvg'] = vol_sum / 20
            data[i]['volRatio'] = data[i]['volume'] / (vol_sum / 20) if vol_sum > 0 else 0

    # MACD (12/26/9)
    def ema(arr, span):
        k = 2 / (span + 1)
        out = [arr[0]]
        for i in range(1, len(arr)):
            out.append(arr[i] * k + out[i - 1] * (1 - k))
        return out

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
    macd_signal = ema(macd_line, 9)
    for i in range(n):
        data[i]['macd'] = macd_line[i]
        data[i]['macdSignal'] = macd_signal[i]
        data[i]['macdHist'] = macd_line[i] - macd_signal[i]

    # 캔들스틱 패턴
    data[0]['isHammer'] = False
    data[0]['isBullishEngulfing'] = False
    for i in range(1, n):
        d, p = data[i], data[i - 1]
        body = d['close'] - d['open']
        body_abs = abs(body)
        upper = d['high'] - max(d['open'], d['close'])
        lower = min(d['open'], d['close']) - d['low']
        d['isHammer'] = lower > 2 * body_abs and upper < body_abs * 0.5 and body_abs > 0
        d['isBullishEngulfing'] = (body > 0 and (p['close'] - p['open']) < 0
                                   and d['open'] <= p['close'] and d['close'] >= p['open'])

    return data


# ============================================================
# 3. 벤포드 법칙 (JS 1:1 포팅)
# ============================================================
def first_digit(n):
    n = abs(n)
    if n == 0:
        return 0
    while n < 1:
        n *= 10
    while n >= 10:
        n /= 10
    return int(n)


def benford_chi2(values):
    digits = [first_digit(v) for v in values if v != 0]
    digits = [d for d in digits if 1 <= d <= 9]
    if len(digits) < 5:
        return 0
    observed = [0] * 9
    for d in digits:
        observed[d - 1] += 1
    chi2 = 0
    for d in range(1, 10):
        exp = BENFORD[d] * len(digits)
        if exp > 0:
            chi2 += (observed[d - 1] - exp) ** 2 / exp
    return chi2


def benford_score(values):
    chi2 = benford_chi2(values)
    return 1 - 1 / (1 + chi2 / 15)


def precompute_benford_daily_scores(data, benford_window):
    scores = [0.0] * len(data)
    for idx in range(benford_window, len(data)):
        vol_slice = [data[j]['volume'] for j in range(max(0, idx - benford_window), idx + 1)]
        vbs = benford_score(vol_slice)
        price_slice = [data[j]['close'] for j in range(max(0, idx - benford_window), idx + 1)]
        changes = [abs(price_slice[j] - price_slice[j - 1]) for j in range(1, len(price_slice))]
        pbs = benford_score([c for c in changes if c > 0])
        scores[idx] = (vbs + pbs) * 0.1
    return scores


# ============================================================
# 4. 매수 스코어 계산 (JS calcBuyScore 1:1 포팅)
# ============================================================
def calc_buy_score(data, idx, benford_influence=15, benford_window=30,
                   benford_min_hits=3, benford_daily_scores=None, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if idx < 60:
        return 0
    row = data[idx]
    prev = data[idx - 1]
    ma5, ma20, ma60 = row.get('ma5'), row.get('ma20'), row.get('ma60')
    if ma5 is None or ma20 is None or ma60 is None:
        return 0

    # 절대 차단
    if row.get('volRatio') is not None and row['volRatio'] > 10:
        return 0
    if row.get('rsi') is not None and row['rsi'] > 85:
        return 0

    score = 0

    # ══ PATTERN A: 과매도 반등 ══
    if row.get('rsi') is not None:
        rsi = row['rsi']
        if rsi < 25:
            score += weights['rsi_extreme_oversold']
        elif rsi < 35:
            score += weights['rsi_oversold']
        elif rsi < 45:
            score += weights['rsi_mild_oversold']
        elif rsi < 55:
            score += weights['rsi_neutral']
        elif rsi < 65:
            score += weights['rsi_momentum']
        elif rsi < 80:
            pass
        else:
            score -= weights['rsi_overbought_penalty']

    # BB 위치
    if row.get('bbMid') is not None and row.get('bbUpper') is not None and row.get('bbLower') is not None:
        bb_range = row['bbUpper'] - row['bbLower']
        bb_pos = (row['close'] - row['bbLower']) / bb_range if bb_range > 0 else 0.5
        if bb_pos < 0.15:
            score += weights['bb_below']
        elif bb_pos < 0.30:
            score += weights['bb_lower']
        elif bb_pos < 0.45:
            score += weights['bb_low']
        elif bb_pos > 0.85:
            score -= weights['bb_high_penalty']

    # 최근 5일 하락
    if idx >= 5:
        ret5 = (row['close'] - data[idx - 5]['close']) / data[idx - 5]['close']
        if ret5 < -0.08:
            score += weights['dip_sharp']
        elif ret5 < -0.04:
            score += weights['dip_moderate']
        elif ret5 < -0.01:
            score += weights['dip_mild']

    # 연속 하락 후 반전
    consec_down = 0
    for lb in range(1, 10):
        if idx - lb < 0:
            break
        if data[idx - lb]['close'] < data[idx - lb]['open']:
            consec_down += 1
        else:
            break
    if consec_down >= 3 and row['close'] > row['open']:
        score += weights['reversal_strong']
    elif consec_down >= 2 and row['close'] > row['open']:
        score += weights['reversal_moderate']
    elif row['close'] > row['open'] and prev['close'] < prev['open']:
        score += weights['reversal_mild']

    # 20일 저점 근접
    low20 = min(data[j]['low'] for j in range(max(0, idx - 20), idx + 1))
    if low20 > 0:
        dist_low = (row['close'] - low20) / low20
        if dist_low < 0.02:
            score += weights['low20_near']
        elif dist_low < 0.05:
            score += weights['low20_vicinity']

    # MA20 이격
    gap_ma20 = (row['close'] - ma20) / ma20
    if gap_ma20 < -0.08:
        score += weights['ma20_gap_large']
    elif gap_ma20 < -0.04:
        score += weights['ma20_gap_moderate']

    # ══ PATTERN B: 추세 추종 ══
    if ma5 > ma20 and ma20 > ma60:
        score += weights['align_full']
    elif ma5 > ma20:
        score += weights['align_partial']

    # MA20 기울기
    if idx >= 10 and data[idx - 10].get('ma20') is not None and data[idx - 10]['ma20'] > 0:
        slope = (ma20 - data[idx - 10]['ma20']) / data[idx - 10]['ma20']
        if slope > 0.03:
            score += weights['slope_steep']
        elif slope > 0.01:
            score += weights['slope_moderate']
        elif slope > 0:
            score += weights['slope_mild']

    # 20일 고점 근접
    high20 = max(data[j]['high'] for j in range(max(0, idx - 20), idx + 1))
    if high20 > 0 and (high20 - row['close']) / high20 < 0.03:
        score += weights['high20_near']

    # ══ 공통 시그널 ══
    if row.get('macdHist') is not None and row['macdHist'] > 0:
        score += 0.8
        if prev.get('macdHist') is not None and prev['macdHist'] <= 0:
            score += weights['macd_golden_cross']
        elif prev.get('macdHist') is not None and row['macdHist'] > prev['macdHist']:
            score += 0.3

    if row.get('volRatio') is not None:
        if 1.2 <= row['volRatio'] <= 3:
            score += weights['vol_healthy']
        elif 3 < row['volRatio'] <= 6:
            score += 0.3
        elif row['volRatio'] > 6:
            score -= 0.5

    if row['close'] > row['open']:
        score += 0.5

    if row.get('isBullishEngulfing'):
        score += weights['engulfing']

    # ══ 리스크 감점 ══
    if idx >= 5:
        ret5 = (row['close'] - data[idx - 5]['close']) / data[idx - 5]['close']
        if ret5 > 0.15:
            score -= weights['surge_risk']

    # ══ 벤포드 승수 ══
    max_influence = (benford_influence or 15) / 100
    bw = benford_window or 30
    min_hits = benford_min_hits or 3
    bmult = 1.0
    if max_influence > 0 and benford_daily_scores is not None:
        hit_count = 0
        total_str = 0
        for d_off in range(bw):
            if idx - d_off < 0:
                break
            ds = benford_daily_scores[idx - d_off]
            if ds > 0.03:
                hit_count += 1
                total_str += ds
        if hit_count >= min_hits:
            hr = hit_count / bw
            a_s = total_str / hit_count
            bmult = 1 + min(hr * a_s * 5, max_influence)
            if hit_count >= min_hits * 2:
                bmult *= 1.05

    return score * bmult


# ============================================================
# 5. 백테스터 (JS runBacktest 1:1 포팅)
# ============================================================
def run_backtest(data, take_profit, stop_loss, cooldown, benford_window=30,
                 benford_influence=15, benford_min_hits=3, buy_threshold=5.0,
                 start_idx=60, commission_rate=0.015, tax_rate=0.18,
                 weights=None, _benford_daily_scores=None):
    comm = commission_rate / 100
    tax = tax_rate / 100
    max_infl = (benford_influence or 15) / 100
    benford_daily_scores = _benford_daily_scores or (
        precompute_benford_daily_scores(data, benford_window) if max_infl > 0 else None
    )
    trades = []
    last_buy_idx = -cooldown
    begin_idx = max(60, start_idx)

    for idx in range(begin_idx, len(data)):
        if idx - last_buy_idx < cooldown:
            continue
        sc = calc_buy_score(data, idx, benford_influence, benford_window,
                            benford_min_hits, benford_daily_scores, weights)
        if sc < buy_threshold:
            continue

        entry_price = data[idx]['close']
        target_price = entry_price * (1 + take_profit)
        stop_price = entry_price * (1 - stop_loss)
        result = None
        exit_idx = None
        exit_price = None

        for j in range(idx + 1, len(data)):
            day = data[j]
            hit_stop = day['low'] <= stop_price
            hit_target = day['high'] >= target_price
            if hit_target and hit_stop:
                if day['open'] <= stop_price:
                    result, exit_price = 'LOSS', stop_price
                elif day['open'] >= target_price:
                    result, exit_price = 'WIN', target_price
                else:
                    result, exit_price = 'LOSS', stop_price
                exit_idx = j
                break
            elif hit_target:
                result, exit_price = 'WIN', target_price
                exit_idx = j
                break
            elif hit_stop:
                result, exit_price = 'LOSS', stop_price
                exit_idx = j
                break

        if result is None:
            result = 'OPEN'
            exit_idx = len(data) - 1
            exit_price = data[-1]['close']

        buy_cost = entry_price * (1 + comm)
        sell_revenue = exit_price * (1 - comm - tax)
        ret_pct = (sell_revenue - buy_cost) / buy_cost * 100
        if result != 'OPEN':
            result = 'WIN' if ret_pct >= 0 else 'LOSS'

        trades.append({
            'entry_idx': idx, 'exit_idx': exit_idx,
            'entry_price': entry_price, 'exit_price': round(exit_price),
            'score': round(sc, 1), 'result': result,
            'return_pct': round(ret_pct, 2),
        })
        last_buy_idx = idx
    return trades


def summarize(trades):
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    closed = wins + losses
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_ret = sum(t['return_pct'] for t in closed) / len(closed) if closed else 0
    return {
        'total': len(trades), 'wins': len(wins), 'losses': len(losses),
        'closed': len(closed), 'win_rate': round(win_rate, 1),
        'avg_return': round(avg_ret, 2),
    }


def analyze_volatility(data):
    rets = []
    for i in range(1, len(data)):
        if data[i - 1]['close'] > 0:
            rets.append(abs(data[i]['close'] - data[i - 1]['close']) / data[i - 1]['close'])
    if not rets:
        return 0.03
    atr_sum = 0
    for i in range(1, len(data)):
        tr = max(
            data[i]['high'] - data[i]['low'],
            abs(data[i]['high'] - data[i - 1]['close']),
            abs(data[i]['low'] - data[i - 1]['close'])
        )
        atr_sum += tr / data[i - 1]['close'] if data[i - 1]['close'] > 0 else 0
    return atr_sum / (len(data) - 1) if len(data) > 1 else 0.03


# ============================================================
# 6. 최적화 엔진
# ============================================================
def get_tp_sl_ranges(daily_vol):
    """JS autoOptimize의 변동성 기반 탐색 범위 (1:1)"""
    if daily_vol < 1.5:
        return [0.02, 0.03, 0.04, 0.06, 0.08, 0.10], [0.03, 0.05, 0.07, 0.10]
    elif daily_vol < 2.5:
        return [0.03, 0.04, 0.06, 0.08, 0.10, 0.13], [0.04, 0.06, 0.08, 0.10]
    elif daily_vol < 4.0:
        return [0.04, 0.06, 0.08, 0.10, 0.15, 0.21], [0.05, 0.07, 0.10, 0.13]
    else:
        return [0.05, 0.08, 0.10, 0.15, 0.21, 0.30], [0.07, 0.10, 0.13, 0.15]


def optimize_single_stock(args):
    """단일 종목 파라미터 최적화 (워커 함수)"""
    code, stock_data, weights, param_grid = args
    data = calc_indicators(copy.deepcopy(stock_data['data']))
    n = len(data)
    if n < 120:
        return code, None

    # 워크포워드: 앞 70% 훈련 / 뒤 30% 테스트
    train_end = int(n * 0.7)
    train_data = data[:train_end]
    daily_vol = analyze_volatility(stock_data['data']) * 100
    tp_range, sl_range = get_tp_sl_ranges(daily_vol)

    # 벤포드 캐시
    benford_caches = {}
    for bw in param_grid['bw']:
        benford_caches[bw] = precompute_benford_daily_scores(train_data, bw)

    best_stable = None  # 안정형 (승률)
    best_profit = None  # 수익형 (수익률)

    for tp in tp_range:
        for sl in sl_range:
            for cd in param_grid['cd']:
                for bw in param_grid['bw']:
                    for bi in param_grid['bi']:
                        for bmh in param_grid['bmh']:
                            for thr in param_grid['threshold']:
                                trades = run_backtest(
                                    train_data, tp, sl, cd, bw, bi, bmh,
                                    buy_threshold=thr, weights=weights,
                                    _benford_daily_scores=benford_caches[bw]
                                )
                                s = summarize(trades)
                                if s['closed'] < 3:
                                    continue

                                # 안정형 스코어 (JS 로직 그대로)
                                wr = s['win_rate']
                                win_sc = wr * 0.8 if wr >= 80 else (wr * 0.55 if wr >= 70 else wr * 0.3)
                                trade_sc = min(s['closed'] / 8, 1) * 10
                                penalty = -30 if wr < 50 else (-15 if wr < 60 else 0)
                                stable_score = win_sc + trade_sc + penalty

                                params = {'tp': tp, 'sl': sl, 'cd': cd, 'bw': bw,
                                          'bi': bi, 'bmh': bmh, 'thr': thr}

                                if best_stable is None or stable_score > best_stable['score']:
                                    best_stable = {**params, 'score': stable_score,
                                                   'win_rate': wr, 'trades': s['closed'],
                                                   'avg_return': s['avg_return']}

                                # 수익형 스코어
                                p_score = (s['avg_return'] * 5 +
                                           min(max(0, 0), 200) * 0.05 +
                                           min(s['closed'] / 5, 1) * 3 +
                                           (-30 if wr < 35 else 0))
                                if best_profit is None or p_score > best_profit['score']:
                                    best_profit = {**params, 'score': p_score,
                                                   'win_rate': wr, 'trades': s['closed'],
                                                   'avg_return': s['avg_return']}

    if best_stable is None:
        return code, None

    # 테스트 구간 평가
    for best in [best_stable, best_profit]:
        if best is None:
            continue
        test_trades = run_backtest(
            data, best['tp'], best['sl'], best['cd'], best['bw'],
            best['bi'], best['bmh'], buy_threshold=best['thr'],
            start_idx=train_end, weights=weights
        )
        ts = summarize(test_trades)
        best['test_win_rate'] = ts['win_rate']
        best['test_trades'] = ts['closed']
        best['test_avg_return'] = ts['avg_return']

    return code, {'stable': best_stable, 'profit': best_profit, 'daily_vol': daily_vol}


def phase1_optimize(stocks, weights=None):
    """Phase 1: 전 종목 파라미터 최적화"""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    param_grid = {
        'cd': [2, 3, 5, 7],
        'bw': [15, 20, 30],
        'bi': [5, 10, 15, 20],
        'bmh': [2, 3, 5],
        'threshold': [4.5, 5.0, 5.5, 6.0],
    }

    args_list = [(code, sdata, weights, param_grid) for code, sdata in stocks.items()]
    results = {}
    total = len(args_list)

    # 병렬 처리
    workers = max(1, cpu_count() - 1)
    print(f'  병렬 처리: {workers} 워커, {total} 종목')

    with Pool(workers) as pool:
        for i, (code, result) in enumerate(pool.imap_unordered(optimize_single_stock, args_list)):
            name = STOCK_CODES.get(code, code)
            if result:
                results[code] = result
                wr = result['stable']['win_rate']
                twr = result['stable']['test_win_rate']
                sys.stdout.write(f'\r  [{i+1}/{total}] {name}: 훈련 {wr:.1f}% / 테스트 {twr:.1f}%' + ' ' * 20)
            else:
                sys.stdout.write(f'\r  [{i+1}/{total}] {name}: 데이터 부족, 스킵' + ' ' * 20)
            sys.stdout.flush()

    print(f'\r  Phase 1 완료: {len(results)}/{total} 종목 분석 성공' + ' ' * 40)
    return results


def aggregate_results(stock_results, strategy='stable'):
    """전 종목 결과 집계"""
    total_wins = 0
    total_losses = 0
    win_rates = []
    avg_returns = []
    test_win_rates = []
    test_avg_returns = []

    for code, res in stock_results.items():
        best = res.get(strategy)
        if best is None:
            continue
        total_wins += int(best['trades'] * best['win_rate'] / 100)
        total_losses += best['trades'] - int(best['trades'] * best['win_rate'] / 100)
        win_rates.append(best['win_rate'])
        avg_returns.append(best['avg_return'])
        if best.get('test_trades', 0) > 0:
            test_win_rates.append(best['test_win_rate'])
            test_avg_returns.append(best['test_avg_return'])

    total_trades = total_wins + total_losses
    return {
        'portfolio_win_rate': round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
        'avg_stock_win_rate': round(sum(win_rates) / len(win_rates), 1) if win_rates else 0,
        'median_win_rate': round(sorted(win_rates)[len(win_rates) // 2], 1) if win_rates else 0,
        'avg_return': round(sum(avg_returns) / len(avg_returns), 2) if avg_returns else 0,
        'consistency_60': round(sum(1 for wr in win_rates if wr >= 60) / len(win_rates) * 100, 1) if win_rates else 0,
        'total_trades': total_trades,
        'stocks_analyzed': len(win_rates),
        # 테스트 구간
        'test_portfolio_wr': round(sum(test_win_rates) / len(test_win_rates), 1) if test_win_rates else 0,
        'test_avg_return': round(sum(test_avg_returns) / len(test_avg_returns), 2) if test_avg_returns else 0,
        'test_stocks': len(test_win_rates),
    }


def phase2_weight_optimize(stocks, base_results, iterations=300):
    """Phase 2: 스코어링 가중치 최적화 (coordinate descent)"""
    print(f'\n  가중치 최적화 시작 ({iterations} 반복)...')
    best_weights = copy.deepcopy(DEFAULT_WEIGHTS)

    # 기준선: 기본 가중치 성과
    base_agg = aggregate_results(base_results, 'stable')
    best_score = base_agg['avg_stock_win_rate']
    print(f'  기준 승률: {best_score:.1f}%')

    # 최적화할 가중치 그룹
    weight_groups = [
        # 그룹 1: RSI (가장 영향 큰 항목)
        ['rsi_extreme_oversold', 'rsi_oversold', 'rsi_mild_oversold'],
        # 그룹 2: BB + Dip
        ['bb_below', 'bb_lower', 'dip_sharp', 'dip_moderate'],
        # 그룹 3: 추세
        ['align_full', 'slope_steep', 'high20_near'],
        # 그룹 4: 반전 + 기타
        ['reversal_strong', 'reversal_moderate', 'low20_near', 'ma20_gap_large'],
        # 그룹 5: 공통
        ['macd_golden_cross', 'vol_healthy', 'engulfing', 'surge_risk'],
    ]

    # 빠른 평가를 위해 종목 수 제한 (상위 50개)
    sample_codes = list(stocks.keys())[:50]
    sample_stocks = {c: stocks[c] for c in sample_codes if c in stocks}

    param_grid_small = {
        'cd': [3, 5],
        'bw': [20, 30],
        'bi': [10, 15],
        'bmh': [3],
        'threshold': [5.0],
    }

    improved = False
    for group_idx, group in enumerate(weight_groups):
        print(f'  그룹 {group_idx + 1}/{len(weight_groups)}: {group}')
        # 각 가중치에 대해 ±20%, ±40% 테스트
        for key in group:
            original = best_weights[key]
            candidates = [
                round(original * 0.6, 2),
                round(original * 0.8, 2),
                original,
                round(original * 1.2, 2),
                round(original * 1.4, 2),
            ]
            candidates = list(set(c for c in candidates if c >= 0))
            candidates.sort()

            for val in candidates:
                if val == original:
                    continue
                test_weights = copy.deepcopy(best_weights)
                test_weights[key] = val

                # 샘플 종목으로 빠른 평가
                args = [(c, s, test_weights, param_grid_small) for c, s in sample_stocks.items()]
                test_results = {}
                with Pool(max(1, cpu_count() - 1)) as pool:
                    for code, result in pool.imap_unordered(optimize_single_stock, args):
                        if result:
                            test_results[code] = result

                agg = aggregate_results(test_results, 'stable')
                if agg['avg_stock_win_rate'] > best_score:
                    best_score = agg['avg_stock_win_rate']
                    best_weights[key] = val
                    improved = True
                    print(f'    {key}: {original} → {val} (승률 {best_score:.1f}%)')

    if not improved:
        print('  기본 가중치가 이미 최적입니다.')

    return best_weights, best_score


# ============================================================
# 7. 리포트 생성
# ============================================================
def generate_report(stock_results, optimized_weights, base_agg, opt_agg, stocks):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append('=' * 76)
    lines.append('  대규모 종목 분석 최적화 리포트')
    lines.append(f'  생성: {timestamp}')
    lines.append(f'  분석 종목: {base_agg["stocks_analyzed"]}개')
    lines.append('=' * 76)

    # 기본값 vs 최적값
    lines.append('\n■ 기본값 vs 최적값 비교 (안정형/승률 기준)')
    lines.append('─' * 76)
    lines.append(f'  {"항목":<24} {"기본값":>12} {"최적값":>12} {"변화":>12}')
    lines.append('─' * 76)
    lines.append(f'  {"포트폴리오 승률":<24} {base_agg["portfolio_win_rate"]:>11.1f}% {opt_agg["portfolio_win_rate"]:>11.1f}% {opt_agg["portfolio_win_rate"]-base_agg["portfolio_win_rate"]:>+11.1f}%')
    lines.append(f'  {"종목별 평균 승률":<24} {base_agg["avg_stock_win_rate"]:>11.1f}% {opt_agg["avg_stock_win_rate"]:>11.1f}% {opt_agg["avg_stock_win_rate"]-base_agg["avg_stock_win_rate"]:>+11.1f}%')
    lines.append(f'  {"중앙값 승률":<24} {base_agg["median_win_rate"]:>11.1f}% {opt_agg["median_win_rate"]:>11.1f}% {opt_agg["median_win_rate"]-base_agg["median_win_rate"]:>+11.1f}%')
    lines.append(f'  {"평균 수익률":<24} {base_agg["avg_return"]:>+11.2f}% {opt_agg["avg_return"]:>+11.2f}% {opt_agg["avg_return"]-base_agg["avg_return"]:>+11.2f}%')
    lines.append(f'  {"일관성 (≥60% 종목)":<24} {base_agg["consistency_60"]:>11.1f}% {opt_agg["consistency_60"]:>11.1f}% {opt_agg["consistency_60"]-base_agg["consistency_60"]:>+11.1f}%')
    lines.append(f'  {"총 거래수":<24} {base_agg["total_trades"]:>12,} {opt_agg["total_trades"]:>12,}')

    # 테스트 구간 (과적합 확인)
    lines.append('\n■ 테스트 구간 성과 (과적합 확인)')
    lines.append('─' * 76)
    lines.append(f'  테스트 평균 승률: {opt_agg["test_portfolio_wr"]:.1f}% (훈련: {opt_agg["avg_stock_win_rate"]:.1f}%)')
    gap = opt_agg['avg_stock_win_rate'] - opt_agg['test_portfolio_wr']
    if gap > 20:
        lines.append(f'  ⚠ 훈련-테스트 갭 {gap:.1f}% — 과적합 위험')
    elif gap > 10:
        lines.append(f'  △ 훈련-테스트 갭 {gap:.1f}% — 약간의 과적합')
    else:
        lines.append(f'  ✓ 훈련-테스트 갭 {gap:.1f}% — 양호')

    # 가중치 변경 내역
    lines.append('\n■ 스코어링 가중치 변경 내역')
    lines.append('─' * 76)
    changes = []
    for key in sorted(DEFAULT_WEIGHTS.keys()):
        old = DEFAULT_WEIGHTS[key]
        new = optimized_weights[key]
        if old != new:
            pct = (new - old) / old * 100 if old != 0 else 0
            changes.append(f'  {key:<30} {old:>6.1f} → {new:>6.1f} ({pct:>+.0f}%)')
    if changes:
        for c in changes:
            lines.append(c)
    else:
        lines.append('  (변경 없음 — 기본 가중치가 최적)')

    # 종목별 상세 (상위/하위)
    lines.append('\n■ 종목별 성과 (안정형, 훈련 승률 상위 20)')
    lines.append('─' * 76)
    lines.append(f'  {"코드":<8} {"종목명":<16} {"훈련WR":>8} {"테스트WR":>8} {"거래":>6} {"TP":>5} {"SL":>5} {"CD":>4}')
    lines.append('─' * 76)

    sorted_results = sorted(
        [(c, r) for c, r in stock_results.items() if r.get('stable')],
        key=lambda x: x[1]['stable']['win_rate'], reverse=True
    )
    for code, res in sorted_results[:20]:
        s = res['stable']
        name = STOCK_CODES.get(code, code)[:8]
        lines.append(
            f'  {code:<8} {name:<16} {s["win_rate"]:>7.1f}% {s.get("test_win_rate", 0):>7.1f}% '
            f'{s["trades"]:>5}건 {s["tp"]*100:>4.0f}% {s["sl"]*100:>4.0f}% {s["cd"]:>3}일'
        )

    # 수익형 상위
    lines.append('\n■ 종목별 성과 (수익형, 평균수익률 상위 20)')
    lines.append('─' * 76)
    sorted_profit = sorted(
        [(c, r) for c, r in stock_results.items() if r.get('profit')],
        key=lambda x: x[1]['profit']['avg_return'], reverse=True
    )
    for code, res in sorted_profit[:20]:
        p = res['profit']
        name = STOCK_CODES.get(code, code)[:8]
        lines.append(
            f'  {code:<8} {name:<16} WR:{p["win_rate"]:>5.1f}% 수익률:{p["avg_return"]:>+6.2f}% '
            f'{p["trades"]:>4}건 TP:{p["tp"]*100:.0f}% SL:{p["sl"]*100:.0f}%'
        )

    # 적용 방법
    lines.append('\n■ index.html 적용 방법')
    lines.append('─' * 76)
    lines.append('  index.html의 calcBuyScore 함수에서 아래 값을 변경:')
    for key in sorted(DEFAULT_WEIGHTS.keys()):
        old = DEFAULT_WEIGHTS[key]
        new = optimized_weights[key]
        if old != new:
            lines.append(f'  - {key}: {old} → {new}')
    lines.append('')
    lines.append('  또는, 새 파일(index_optimized.html)을 생성하여 테스트 후 교체를 권장합니다.')
    lines.append('=' * 76)

    report_text = '\n'.join(lines)

    # 파일 저장
    report_path = os.path.join(BASE_DIR, 'optimization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # JSON
    json_data = {
        'generated': timestamp,
        'stocks_analyzed': base_agg['stocks_analyzed'],
        'default_weights': DEFAULT_WEIGHTS,
        'optimized_weights': optimized_weights,
        'base_aggregate': base_agg,
        'optimized_aggregate': opt_agg,
        'per_stock': {
            code: {
                'name': STOCK_CODES.get(code, code),
                'stable': res.get('stable'),
                'profit': res.get('profit'),
                'daily_vol': res.get('daily_vol'),
            }
            for code, res in stock_results.items()
        },
    }
    json_path = os.path.join(BASE_DIR, 'optimization_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return report_text, report_path, json_path


# ============================================================
# 8. 메인
# ============================================================
def main():
    print('=' * 76)
    print('  대규모 종목 분석 최적화 시스템')
    print(f'  종목: {len(STOCK_CODES)}개 / 네이버 증권 API')
    print('=' * 76)

    # 1. 다운로드
    print('\n[1/4] 종목 데이터 다운로드...')
    paths = download_all_stocks(STOCK_CODES)

    # 2. 파싱
    print('\n[2/4] XML 파싱...')
    stocks = {}
    for code, path in paths.items():
        parsed = parse_naver_xml(path)
        if parsed and len(parsed['data']) >= 120:
            stocks[code] = parsed
    print(f'  유효 종목: {len(stocks)}개 (120일봉 이상)')

    if not stocks:
        print('  분석 가능한 종목이 없습니다.')
        return

    # 3. Phase 1: 파라미터 최적화 (기본 가중치)
    print('\n[3/4] Phase 1: 파라미터 최적화 (기본 가중치)...')
    base_results = phase1_optimize(stocks, DEFAULT_WEIGHTS)
    base_agg = aggregate_results(base_results, 'stable')
    profit_agg = aggregate_results(base_results, 'profit')
    print(f'\n  ── 기본 가중치 결과 ──')
    print(f'  안정형 — 포트폴리오 승률: {base_agg["portfolio_win_rate"]:.1f}%, '
          f'종목평균: {base_agg["avg_stock_win_rate"]:.1f}%, '
          f'일관성: {base_agg["consistency_60"]:.1f}%')
    print(f'  수익형 — 평균수익률: {profit_agg["avg_return"]:+.2f}%, '
          f'종목평균 승률: {profit_agg["avg_stock_win_rate"]:.1f}%')

    # 4. Phase 2: 가중치 최적화
    print('\n[4/4] Phase 2: 스코어링 가중치 최적화...')
    optimized_weights, opt_score = phase2_weight_optimize(stocks, base_results)

    # 최적 가중치로 다시 전체 평가
    if optimized_weights != DEFAULT_WEIGHTS:
        print('\n  최적 가중치로 전체 종목 재평가...')
        opt_results = phase1_optimize(stocks, optimized_weights)
        opt_agg = aggregate_results(opt_results, 'stable')
    else:
        opt_results = base_results
        opt_agg = base_agg

    # 리포트 생성
    print('\n리포트 생성...')
    report_text, report_path, json_path = generate_report(
        opt_results, optimized_weights, base_agg, opt_agg, stocks
    )
    print(report_text)
    print(f'\n저장 완료:')
    print(f'  텍스트: {report_path}')
    print(f'  JSON:   {json_path}')


if __name__ == '__main__':
    main()
