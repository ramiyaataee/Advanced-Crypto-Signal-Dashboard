import streamlit as st
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import schedule
from io import StringIO
import hashlib

# Page configuration
st.set_page_config(
    page_title="Advanced Crypto Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default configuration with improved parameters
DEFAULT_CONFIG = {
    'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT'],
    'min_volume_usd': 1000000,
    'rr_min': 1.5,
    'binance_api': 'https://api.binance.com/api/v3',
    'fear_greed_api': 'https://api.alternative.me/fng/',
    'telegram_enabled': False,
    'telegram_token': '',
    'telegram_chat_id': '',
    'auto_signal_interval': 15,  # minutes
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'ma_short': 20,
    'ma_long': 50,
    'atr_multiplier': 2.0,
    'confidence_threshold': 70,
    'max_signals_per_hour': 3,
    'min_score_threshold': 5,
    'duplicate_signal_window': 1800  # 30 minutes
}

@dataclass
class Signal:
    symbol: str
    side: str
    entry: float
    stop: float
    targets: List[float]
    confidence: float
    rationale: str
    technical_analysis: Dict[str, Any]
    timestamp: float = time.time()
    metadata: Dict[str, Any] = None
    signal_id: str = ""
    
    def __post_init__(self):
        """Generate unique signal ID after initialization"""
        if not self.signal_id:
            signal_data = f"{self.symbol}_{self.side}_{self.entry}_{self.timestamp}"
            self.signal_id = hashlib.md5(signal_data.encode()).hexdigest()[:8]

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, message: str) -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'  # Changed to HTML for better formatting
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_signal(self, signal: Signal) -> str:
        side_emoji = "🔥" if signal.side == "BUY" else "📢"
        direction = "BUY" if signal.side == "BUY" else "SELL"
        
        # Improved HTML formatting
        message = f"""
<b>{side_emoji} {direction} SIGNAL | {signal.symbol}</b>
<b>Signal ID:</b> #{signal.signal_id}

💰 <b>Entry:</b> ${signal.entry:,.4f}
🛑 <b>Stop Loss:</b> ${signal.stop:,.4f}
🎯 <b>Target 1:</b> ${signal.targets[0]:,.4f}"""
        
        if len(signal.targets) > 1:
            message += f"\n🎯 <b>Target 2:</b> ${signal.targets[1]:,.4f}"
        
        risk_amount = abs(signal.entry - signal.stop)
        reward_amount = abs(signal.targets[0] - signal.entry)
        potential_profit = (reward_amount / signal.entry) * 100
        
        message += f"""

⚡ <b>Risk/Reward:</b> {signal.technical_analysis.get('rr_ratio', 0):.1f}
📊 <b>Confidence:</b> {signal.confidence:.0f}%
💹 <b>Potential Profit:</b> {potential_profit:.1f}%

📉 <b>RSI:</b> {signal.technical_analysis.get('rsi', 0):.1f}
📈 <b>MACD:</b> {"Bullish" if signal.technical_analysis.get('macd', 0) > 0 else "Bearish"}
📊 <b>Trend:</b> {signal.technical_analysis.get('trend', 'Neutral')}

⏰ <b>Time:</b> {datetime.fromtimestamp(signal.timestamp).strftime('%Y-%m-%d %H:%M:%S')}

<i>💡 {signal.rationale}</i>
"""
        return message

# Enhanced caching with better error handling
@st.cache_data(ttl=60)
def fetch_ticker(symbol: str) -> Dict[str, Any]:
    try:
        binance_symbol = symbol.replace('/', '')
        response = requests.get(
            f"{DEFAULT_CONFIG['binance_api']}/ticker/24hr",
            params={'symbol': binance_symbol},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                'price': float(data['lastPrice']),
                'volume_24h': float(data['quoteVolume']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'change': float(data['priceChange']),
                'percentage': float(data['priceChangePercent'])
            }
        else:
            st.warning(f"API Error {response.status_code} for {symbol}")
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
    
    return {'price': 0.0, 'volume_24h': 0.0, 'high': 0.0, 'low': 0.0, 'change': 0.0, 'percentage': 0.0}

@st.cache_data(ttl=300)
def fetch_klines(symbol: str, interval: str = '1h', limit: int = 200) -> List[Dict[str, Any]]:
    try:
        binance_symbol = symbol.replace('/', '')
        response = requests.get(
            f"{DEFAULT_CONFIG['binance_api']}/klines",
            params={
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            klines = []
            
            for candle in data:
                klines.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            return klines
        else:
            st.warning(f"API Error {response.status_code} for klines {symbol}")
    except Exception as e:
        st.error(f"Error fetching klines for {symbol}: {e}")
    
    return []

@st.cache_data(ttl=1800)
def fetch_fear_greed_index() -> Dict[str, Any]:
    try:
        response = requests.get(DEFAULT_CONFIG['fear_greed_api'], timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification']
            }
    except Exception as e:
        st.error(f"Error fetching Fear & Greed Index: {e}")
    
    return {'value': 50, 'classification': 'Neutral'}

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Enhanced RSI calculation with better error handling"""
    if len(prices) < period + 1:
        return 50.0
    
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return max(0, min(100, rsi))  # Ensure RSI is within valid range
    except Exception as e:
        st.error(f"Error calculating RSI: {e}")
        return 50.0

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Enhanced MACD calculation"""
    if len(prices) < slow:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    try:
        prices_series = pd.Series(prices)
        exp1 = prices_series.ewm(span=fast).mean()
        exp2 = prices_series.ewm(span=slow).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
    except Exception as e:
        st.error(f"Error calculating MACD: {e}")
        return {'macd': 0, 'signal': 0, 'histogram': 0}

def calculate_moving_averages(prices: List[float], short_period: int = 20, long_period: int = 50) -> Dict[str, float]:
    """Enhanced moving averages calculation"""
    if len(prices) < long_period:
        return {'ma_short': 0, 'ma_long': 0}
    
    try:
        prices_series = pd.Series(prices)
        return {
            'ma_short': float(prices_series.rolling(window=short_period).mean().iloc[-1]),
            'ma_long': float(prices_series.rolling(window=long_period).mean().iloc[-1])
        }
    except Exception as e:
        st.error(f"Error calculating moving averages: {e}")
        return {'ma_short': 0, 'ma_long': 0}

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands for additional confirmation"""
    if len(prices) < period:
        return {'upper': 0, 'middle': 0, 'lower': 0}
    
    try:
        prices_series = pd.Series(prices)
        middle = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': float(upper.iloc[-1]),
            'middle': float(middle.iloc[-1]),
            'lower': float(lower.iloc[-1])
        }
    except Exception as e:
        st.error(f"Error calculating Bollinger Bands: {e}")
        return {'upper': 0, 'middle': 0, 'lower': 0}

def analyze_symbol(symbol: str, config: Dict) -> Optional[Signal]:
    """Enhanced signal analysis with multiple confirmations"""
    try:
        klines = fetch_klines(symbol, '1h', 200)
        if len(klines) < config['ma_long']:
            return None
        
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        # Calculate all technical indicators
        rsi = calculate_rsi(closes, config['rsi_period'])
        macd = calculate_macd(closes, config['macd_fast'], config['macd_slow'], config['macd_signal'])
        ma = calculate_moving_averages(closes, config['ma_short'], config['ma_long'])
        bb = calculate_bollinger_bands(closes)
        
        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) > 1 else current_price
        
        # Market data
        ticker = fetch_ticker(symbol)
        fear_greed = fetch_fear_greed_index()
        
        # Calculate ATR (Average True Range)
        true_ranges = []
        for i in range(1, len(klines)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else current_price * 0.02
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_spike = current_volume > avg_volume * 1.5
        
        # Enhanced buy conditions with multiple confirmations
        buy_conditions = [
            ma['ma_short'] > ma['ma_long'] and ma['ma_short'] > 0 and ma['ma_long'] > 0,  # Trend
            config['rsi_oversold'] < rsi < config['rsi_overbought'],  # RSI in range
            macd['macd'] > macd['signal'],  # MACD bullish
            macd['histogram'] > 0,  # MACD histogram positive
            ticker['volume_24h'] > config['min_volume_usd'],  # Volume requirement
            current_price > prev_price * 0.995,  # Price momentum
            fear_greed['value'] < 80,  # Not extreme greed
            current_price > bb['lower'] if bb['lower'] > 0 else True,  # Above BB lower
            volume_spike,  # Volume confirmation
            rsi > 35  # Not oversold
        ]
        
        # Enhanced sell conditions
        sell_conditions = [
            ma['ma_short'] < ma['ma_long'] and ma['ma_short'] > 0 and ma['ma_long'] > 0,  # Trend
            config['rsi_oversold'] < rsi < config['rsi_overbought'],  # RSI in range
            macd['macd'] < macd['signal'],  # MACD bearish
            macd['histogram'] < 0,  # MACD histogram negative
            ticker['volume_24h'] > config['min_volume_usd'],  # Volume requirement
            current_price < prev_price * 1.005,  # Price momentum
            fear_greed['value'] > 20,  # Not extreme fear
            current_price < bb['upper'] if bb['upper'] > 0 else True,  # Below BB upper
            volume_spike,  # Volume confirmation
            rsi < 65  # Not overbought
        ]
        
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        # Generate signals based on improved logic
        if buy_score >= config['min_score_threshold']:
            entry = current_price
            stop_loss = max(current_price - (atr * config['atr_multiplier']), bb['lower'] if bb['lower'] > 0 else current_price * 0.95)
            target1 = min(current_price + (atr * 3), bb['upper'] if bb['upper'] > 0 else current_price * 1.05)
            target2 = current_price + (atr * 5)
            confidence = min(95, 40 + (buy_score * 6))
            
            risk = abs(entry - stop_loss)
            reward = abs(target1 - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio >= config['rr_min'] and confidence >= config['confidence_threshold']:
                return Signal(
                    symbol=symbol,
                    side='BUY',
                    entry=entry,
                    stop=stop_loss,
                    targets=[target1, target2],
                    confidence=confidence,
                    rationale=f"BUY Signal - RSI: {rsi:.1f}, MACD Bullish, Uptrend confirmed, Volume spike",
                    technical_analysis={
                        'rsi': round(rsi, 2),
                        'macd': round(macd['macd'], 6),
                        'ma_short': round(ma['ma_short'], 2),
                        'ma_long': round(ma['ma_long'], 2),
                        'atr': round(atr, 2),
                        'rr_ratio': round(rr_ratio, 2),
                        'trend': 'Bullish',
                        'bb_upper': round(bb['upper'], 2),
                        'bb_lower': round(bb['lower'], 2),
                        'volume_spike': volume_spike
                    },
                    metadata={
                        'score': buy_score,
                        'fear_greed': fear_greed,
                        'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'conditions_met': buy_score,
                        'total_conditions': len(buy_conditions)
                    }
                )
        
        elif sell_score >= config['min_score_threshold']:
            entry = current_price
            stop_loss = min(current_price + (atr * config['atr_multiplier']), bb['upper'] if bb['upper'] > 0 else current_price * 1.05)
            target1 = max(current_price - (atr * 3), bb['lower'] if bb['lower'] > 0 else current_price * 0.95)
            target2 = current_price - (atr * 5)
            confidence = min(95, 40 + (sell_score * 6))
            
            risk = abs(entry - stop_loss)
            reward = abs(entry - target1)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio >= config['rr_min'] and confidence >= config['confidence_threshold']:
                return Signal(
                    symbol=symbol,
                    side='SELL',
                    entry=entry,
                    stop=stop_loss,
                    targets=[target1, target2],
                    confidence=confidence,
                    rationale=f"SELL Signal - RSI: {rsi:.1f}, MACD Bearish, Downtrend confirmed, Volume spike",
                    technical_analysis={
                        'rsi': round(rsi, 2),
                        'macd': round(macd['macd'], 6),
                        'ma_short': round(ma['ma_short'], 2),
                        'ma_long': round(ma['ma_long'], 2),
                        'atr': round(atr, 2),
                        'rr_ratio': round(rr_ratio, 2),
                        'trend': 'Bearish',
                        'bb_upper': round(bb['upper'], 2),
                        'bb_lower': round(bb['lower'], 2),
                        'volume_spike': volume_spike
                    },
                    metadata={
                        'score': sell_score,
                        'fear_greed': fear_greed,
                        'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'conditions_met': sell_score,
                        'total_conditions': len(sell_conditions)
                    }
                )
        
        return None
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {e}")
        return None

def is_duplicate_signal(signal: Signal, existing_signals: List[Dict], window_seconds: int = 1800) -> bool:
    """Check if signal is duplicate within time window"""
    current_time = signal.timestamp
    
    for existing in existing_signals:
        if (existing['symbol'] == signal.symbol and 
            existing['side'] == signal.side and
            abs(existing['timestamp'] - current_time) < window_seconds):
            # Check if price is similar (within 1%)
            price_diff = abs(existing.get('entry', 0) - signal.entry) / signal.entry
            if price_diff < 0.01:  # 1% threshold
                return True
    
    return False

def settings_panel():
    """Comprehensive settings panel"""
    st.sidebar.header("⚙️ System Settings")
    
    # General Settings
    with st.sidebar.expander("🔧 General Settings", expanded=True):
        symbols = st.multiselect(
            "Trading Pairs:",
            ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT', 
             'LINK/USDT', 'MATIC/USDT', 'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'XRP/USDT'],
            default=DEFAULT_CONFIG['symbols']
        )
        
        min_volume = st.number_input(
            "Minimum 24h Volume (USD):",
            min_value=100000,
            max_value=100000000,
            value=DEFAULT_CONFIG['min_volume_usd'],
            step=100000
        )
        
        rr_min = st.slider(
            "Minimum Risk/Reward Ratio:",
            min_value=1.0,
            max_value=5.0,
            value=DEFAULT_CONFIG['rr_min'],
            step=0.1
        )
        
        confidence_threshold = st.slider(
            "Minimum Confidence %:",
            min_value=50,
            max_value=95,
            value=DEFAULT_CONFIG['confidence_threshold'],
            step=5
        )
        
        min_score_threshold = st.slider(
            "Minimum Signal Score:",
            min_value=3,
            max_value=8,
            value=DEFAULT_CONFIG['min_score_threshold'],
            step=1,
            help="Minimum number of conditions that must be met"
        )
    
    # Technical Indicators
    with st.sidebar.expander("📊 Technical Indicators"):
        rsi_period = st.number_input("RSI Period:", min_value=5, max_value=50, value=DEFAULT_CONFIG['rsi_period'])
        rsi_oversold = st.number_input("RSI Oversold Level:", min_value=20, max_value=40, value=DEFAULT_CONFIG['rsi_oversold'])
        rsi_overbought = st.number_input("RSI Overbought Level:", min_value=60, max_value=90, value=DEFAULT_CONFIG['rsi_overbought'])
        
        macd_fast = st.number_input("MACD Fast:", min_value=5, max_value=20, value=DEFAULT_CONFIG['macd_fast'])
        macd_slow = st.number_input("MACD Slow:", min_value=20, max_value=40, value=DEFAULT_CONFIG['macd_slow'])
        macd_signal = st.number_input("MACD Signal:", min_value=5, max_value=15, value=DEFAULT_CONFIG['macd_signal'])
        
        ma_short = st.number_input("Short MA Period:", min_value=5, max_value=50, value=DEFAULT_CONFIG['ma_short'])
        ma_long = st.number_input("Long MA Period:", min_value=30, max_value=200, value=DEFAULT_CONFIG['ma_long'])
        
        atr_multiplier = st.slider("ATR Multiplier:", min_value=1.0, max_value=5.0, value=DEFAULT_CONFIG['atr_multiplier'], step=0.1)
    
    # Telegram Settings
    with st.sidebar.expander("📱 Telegram Settings"):
        telegram_enabled = st.checkbox("Enable Telegram Notifications", value=DEFAULT_CONFIG['telegram_enabled'])
        
        telegram_token = st.text_input(
            "Bot Token:",
            value=DEFAULT_CONFIG['telegram_token'],
            type="password",
            help="Get your bot token from @BotFather"
        )
        
        telegram_chat_id = st.text_input(
            "Chat ID:",
            value=DEFAULT_CONFIG['telegram_chat_id'],
            help="Get your chat ID from @userinfobot"
        )
        
        if telegram_enabled and telegram_token and telegram_chat_id:
            if st.button("🧪 Test Telegram Connection"):
                bot = TelegramBot(telegram_token, telegram_chat_id)
                test_message = """
<b>✅ Connection Test Successful!</b>
🤖 <b>Crypto Signal Bot is Ready</b>

<i>This is a test message to verify your Telegram integration is working properly.</i>
"""
                if bot.send_message(test_message):
                    st.success("✅ Test message sent successfully!")
                else:
                    st.error("❌ Failed to send test message. Please check your settings.")
    
    return {
        'symbols': symbols,
        'min_volume_usd': min_volume,
        'rr_min': rr_min,
        'confidence_threshold': confidence_threshold,
        'min_score_threshold': min_score_threshold,
        'rsi_period': rsi_period,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal': macd_signal,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'atr_multiplier': atr_multiplier,
        'telegram_enabled': telegram_enabled,
        'telegram_token': telegram_token,
        'telegram_chat_id': telegram_chat_id,
        'binance_api': DEFAULT_CONFIG['binance_api'],
        'fear_greed_api': DEFAULT_CONFIG['fear_greed_api'],
        'duplicate_signal_window': DEFAULT_CONFIG['duplicate_signal_window']
    }

def display_market_overview(config: Dict):
    """Enhanced market overview"""
    st.subheader("📊 Market Overview")
    
    fear_greed = fetch_fear_greed_index()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Color coding based on value
        if fear_greed['value'] < 25:
            color = "🟢"  # Green for extreme fear (buying opportunity)
        elif fear_greed['value'] < 45:
            color = "🟡"  # Yellow for fear
        elif fear_greed['value'] < 55:
            color = "⚪"  # White for neutral
        elif fear_greed['value'] < 75:
            color = "🟠"  # Orange for greed
        else:
            color = "🔴"  # Red for extreme greed (sell signal)
        
        st.metric(
            label="Fear & Greed Index",
            value=f"{color} {fear_greed['value']}",
            delta=fear_greed['classification']
        )
    
    # Market statistics
    total_volume = 0
    positive_change = 0
    total_symbols = min(len(config['symbols']), 8)  # Limit to prevent API overload
    
    for symbol in config['symbols'][:total_symbols]:
        ticker = fetch_ticker(symbol)
        total_volume += ticker['volume_24h']
        
        if ticker['percentage'] > 0:
            positive_change += 1
    
    with col2:
        st.metric(
            label="Total 24h Volume",
            value=f"${total_volume/1000000:.1f}M"
        )
    
    with col3:
        positive_percentage = (positive_change / total_symbols) * 100
        st.metric(
            label="Positive Symbols",
            value=f"{positive_change}/{total_symbols}",
            delta=f"{positive_percentage:.0f}%"
        )
    
    with col4:
        current_time = datetime.now()
        st.metric(
            label="Last Update",
            value=current_time.strftime("%H:%M:%S")
        )

def main():
    """Enhanced main function"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f1f1f 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-positive {
        background: rgba(0, 255, 136, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
    }
    
    .metric-negative {
        background: rgba(255, 68, 68, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .signal-table {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Crypto Signal Dashboard</h1>
        <p>Intelligent Analysis & Automated Signal Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    config = settings_panel()
    
    # Connection status
    if config['telegram_enabled']:
        if config['telegram_token'] and config['telegram_chat_id']:
            st.sidebar.success("✅ Telegram Connected")
        else:
            st.sidebar.error("❌ Telegram Not Configured")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Analysis", 
        "🤖 Auto Scanner", 
        "📈 Advanced Charts", 
        "📋 Signal History", 
        "⚙️ System Status"
    ])
    
    with tab1:
        st.header("📊 Manual Analysis")
        
        display_market_overview(config)
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            selected_symbol = st.selectbox(
                "Select Symbol for Analysis:",
                config['symbols']
            )
            
            if st.button("🔍 Analyze Now", key="manual_analysis"):
                with st.spinner(f"Analyzing {selected_symbol}..."):
                    signal = analyze_symbol(selected_symbol, config)
                    
                    if signal:
                        display_signal_details(signal)
                    else:
                        st.warning("No valid signal found for current market conditions")
                        
                        # Show why no signal was generated
                        st.info("💡 **Analysis Summary:**")
                        klines = fetch_klines(selected_symbol, '1h', 200)
                        if klines:
                            closes = [k['close'] for k in klines]
                            rsi = calculate_rsi(closes, config['rsi_period'])
                            macd = calculate_macd(closes, config['macd_fast'], config['macd_slow'], config['macd_signal'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**RSI:** {rsi:.1f}")
                            with col2:
                                st.write(f"**MACD:** {'Bullish' if macd['macd'] > macd['signal'] else 'Bearish'}")
                            with col3:
                                ticker = fetch_ticker(selected_symbol)
                                st.write(f"**24h Change:** {ticker['percentage']:.2f}%")
        
        with analysis_col2:
            # Current price display
            ticker = fetch_ticker(selected_symbol)
            if ticker['price'] > 0:
                price_change_color = "metric-positive" if ticker['percentage'] > 0 else "metric-negative"
                
                st.markdown(f"""
                <div class="{price_change_color}">
                    <h3>{selected_symbol}</h3>
                    <h2>${ticker['price']:,.4f}</h2>
                    <p>{ticker['percentage']:+.2f}% (24h)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric(
                    label="24h Volume",
                    value=f"${ticker['volume_24h']/1000000:.1f}M"
                )
    
    with tab2:
        st.header("🤖 Automated Signal Scanner")
        
        scan_col1, scan_col2 = st.columns([3, 1])
        
        with scan_col1:
            if st.button("🚀 Start Full Scan", key="full_scan"):
                signals = run_full_scan(config)
                
                if signals:
                    st.success(f"✅ Found {len(signals)} new signals!")
                    display_signals_table(signals)
                    
                    # Export options
                    if st.button("💾 Export Signals"):
                        export_signals_to_json(signals)
                else:
                    st.info("No new signals found in current scan")
        
        with scan_col2:
            st.write("**⚡ Quick Actions:**")
            
            if st.button("📊 Market Overview"):
                display_quick_market_overview(config)
            
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        # Auto-scan settings
        st.subheader("⏰ Auto-Scan Configuration")
        
        auto_col1, auto_col2, auto_col3 = st.columns(3)
        
        with auto_col1:
            auto_scan_enabled = st.checkbox(
                "Enable Auto-Scan",
                help="Automatically scan for signals at regular intervals"
            )
        
        with auto_col2:
            scan_interval = st.selectbox(
                "Scan Interval (minutes):",
                [5, 10, 15, 30, 60],
                index=2
            )
        
        with auto_col3:
            max_signals = st.number_input(
                "Max Signals/Hour:",
                min_value=1,
                max_value=10,
                value=3
            )
        
        if auto_scan_enabled:
            st.info(f"🔄 Auto-scan enabled: Every {scan_interval} minutes (Max {max_signals} signals/hour)")
            
            # Initialize session state for auto-scan
            if 'last_scan_time' not in st.session_state:
                st.session_state.last_scan_time = 0
            
            # Check if it's time for auto-scan
            current_time = time.time()
            if current_time - st.session_state.last_scan_time > (scan_interval * 60):
                with st.spinner("🔄 Auto-scanning..."):
                    signals = run_full_scan(config)
                    if signals:
                        st.success(f"🎯 Auto-scan found {len(signals)} signals!")
                        for signal in signals:
                            send_telegram_notification(signal, config)
                    st.session_state.last_scan_time = current_time
    
    with tab3:
        st.header("📈 Advanced Technical Charts")
        
        chart_symbol = st.selectbox(
            "Select Symbol for Chart:",
            config['symbols'],
            key="chart_symbol"
        )
        
        chart_col1, chart_col2 = st.columns([4, 1])
        
        with chart_col2:
            st.subheader("Chart Settings")
            
            chart_interval = st.selectbox(
                "Timeframe:",
                ['1h', '4h', '1d'],
                index=0
            )
            
            chart_periods = st.slider(
                "Candles:",
                min_value=50,
                max_value=500,
                value=200
            )
            
            show_indicators = st.multiselect(
                "Indicators:",
                ['RSI', 'MACD', 'Bollinger Bands', 'Volume'],
                default=['RSI', 'MACD']
            )
            
            if st.button("🔄 Update Chart"):
                st.rerun()
        
        with chart_col1:
            klines = fetch_klines(chart_symbol, chart_interval, chart_periods)
            
            if klines:
                chart = create_advanced_chart(chart_symbol, klines, config, show_indicators)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Technical analysis summary
                display_technical_summary(chart_symbol, klines, config)
            else:
                st.warning("Unable to load chart data")
    
    with tab4:
        st.header("📋 Signal History & Performance")
        
        if 'signal_history' in st.session_state and st.session_state.signal_history:
            display_signal_history()
            display_performance_metrics()
        else:
            st.info("No signal history available yet. Run some scans to populate the history.")
    
    with tab5:
        st.header("⚙️ System Status & Diagnostics")
        
        display_system_status(config)
        display_api_status()
        display_performance_diagnostics()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            🚀 Advanced Crypto Signal Dashboard | Built for Professional Traders
        </div>
        """, 
        unsafe_allow_html=True
    )

def display_signal_details(signal: Signal):
    """Display detailed signal information"""
    st.subheader(f"🎯 Signal: {signal.symbol}")
    
    # Signal header with key info
    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
    
    with signal_col1:
        side_color = "metric-positive" if signal.side == "BUY" else "metric-negative"
        st.markdown(f"""
        <div class="{side_color}">
            <h3>{"🔥" if signal.side == "BUY" else "📢"} {signal.side}</h3>
            <p>Signal ID: #{signal.signal_id}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with signal_col2:
        st.metric("Entry Price", f"${signal.entry:,.4f}")
    
    with signal_col3:
        st.metric("Stop Loss", f"${signal.stop:,.4f}")
        
    with signal_col4:
        risk_amount = abs(signal.entry - signal.stop)
        risk_percent = (risk_amount / signal.entry) * 100
        st.metric("Risk", f"{risk_percent:.2f}%")
    
    # Targets and analysis
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.write("**🎯 Targets:**")
        for i, target in enumerate(signal.targets, 1):
            profit_percent = ((target - signal.entry) / signal.entry * 100) if signal.side == "BUY" else ((signal.entry - target) / signal.entry * 100)
            st.write(f"• Target {i}: ${target:,.4f} (+{profit_percent:.2f}%)")
    
    with detail_col2:
        st.write("**📊 Technical Analysis:**")
        st.write(f"• Confidence: {signal.confidence:.0f}%")
        st.write(f"• R/R Ratio: {signal.technical_analysis.get('rr_ratio', 0):.2f}")
        st.write(f"• RSI: {signal.technical_analysis.get('rsi', 0):.1f}")
        st.write(f"• Trend: {signal.technical_analysis.get('trend', 'N/A')}")
    
    # Rationale
    st.info(f"💡 **Analysis:** {signal.rationale}")

def run_full_scan(config: Dict) -> List[Signal]:
    """Run full market scan"""
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    
    new_signals = []
    progress_bar = st.progress(0)
    total_symbols = len(config['symbols'])
    
    for i, symbol in enumerate(config['symbols']):
        progress_bar.progress((i + 1) / total_symbols)
        
        try:
            signal = analyze_symbol(symbol, config)
            
            if signal and not is_duplicate_signal(signal, st.session_state.signal_history, config['duplicate_signal_window']):
                new_signals.append(signal)
                
                # Add to history
                st.session_state.signal_history.append({
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'entry': signal.entry,
                    'timestamp': signal.timestamp,
                    'signal_id': signal.signal_id,
                    'confidence': signal.confidence
                })
        
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")
    
    progress_bar.empty()
    return new_signals

def display_signals_table(signals: List[Signal]):
    """Enhanced signals table"""
    if not signals:
        st.info("No active signals found")
        return
    
    st.subheader(f"🚨 Active Signals ({len(signals)} found)")
    
    # Prepare data for table
    table_data = []
    for signal in signals:
        profit_potential = ((signal.targets[0] - signal.entry) / signal.entry * 100) if signal.side == 'BUY' else ((signal.entry - signal.targets[0]) / signal.entry * 100)
        
        table_data.append({
            'Symbol': signal.symbol,
            'Side': f"{'🔥' if signal.side == 'BUY' else '📢'} {signal.side}",
            'Entry': f"${signal.entry:,.4f}",
            'Stop': f"${signal.stop:,.4f}",
            'Target': f"${signal.targets[0]:,.4f}",
            'Confidence': f"{signal.confidence:.0f}%",
            'R/R': f"{signal.technical_analysis.get('rr_ratio', 0):.1f}",
            'Potential': f"{profit_potential:.1f}%",
            'Time': datetime.fromtimestamp(signal.timestamp).strftime("%H:%M"),
            'ID': signal.signal_id
        })
    
    df = pd.DataFrame(table_data)
    
    # Color coding for the dataframe
    def highlight_signals(row):
        if 'BUY' in str(row['Side']):
            return ['background-color: rgba(0, 255, 136, 0.1)'] * len(row)
        else:
            return ['background-color: rgba(255, 68, 68, 0.1)'] * len(row)
    
    styled_df = df.style.apply(highlight_signals, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def send_telegram_notification(signal: Signal, config: Dict):
    """Send signal to Telegram if enabled"""
    if config['telegram_enabled'] and config['telegram_token'] and config['telegram_chat_id']:
        bot = TelegramBot(config['telegram_token'], config['telegram_chat_id'])
        message = bot.format_signal(signal)
        
        if bot.send_message(message):
            st.success(f"✅ Signal {signal.signal_id} sent to Telegram")
        else:
            st.error(f"❌ Failed to send signal {signal.signal_id}")

def create_advanced_chart(symbol: str, klines: List[Dict], config: Dict, indicators: List[str]):
    """Create advanced plotly chart with multiple indicators"""
    if not klines:
        return None
    
    df = pd.DataFrame(klines)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    closes = df['close'].tolist()
    
    # Calculate indicators
    rsi_values = []
    macd_values = []
    bb_values = []
    
    for i in range(max(config['rsi_period'], config['macd_slow']), len(closes)):
        if 'RSI' in indicators:
            rsi = calculate_rsi(closes[max(0, i-config['rsi_period']):i+1], config['rsi_period'])
            rsi_values.append(rsi)
        
        if 'MACD' in indicators:
            macd = calculate_macd(closes[max(0, i-config['macd_slow']):i+1], 
                               config['macd_fast'], config['macd_slow'], config['macd_signal'])
            macd_values.append(macd)
        
        if 'Bollinger Bands' in indicators:
            bb = calculate_bollinger_bands(closes[max(0, i-20):i+1])
            bb_values.append(bb)
    
    # Create subplots
    subplot_count = 1 + len([i for i in indicators if i in ['RSI', 'MACD']])
    
    fig = make_subplots(
        rows=subplot_count, 
        cols=1,
        subplot_titles=[f'{symbol} Price'] + [i for i in indicators if i in ['RSI', 'MACD']],
        vertical_spacing=0.05,
        row_heights=[0.6] + [0.2] * (subplot_count - 1)
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Add Bollinger Bands if selected
    if 'Bollinger Bands' in indicators and bb_values:
        start_idx = len(closes) - len(bb_values)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'][start_idx:],
            y=[bb['upper'] for bb in bb_values],
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'][start_idx:],
            y=[bb['lower'] for bb in bb_values],
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255,255,255,0.05)'
        ), row=1, col=1)
    
    # Add RSI if selected
    current_row = 2
    if 'RSI' in indicators and rsi_values:
        start_idx = len(closes) - len(rsi_values)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'][start_idx:],
            y=rsi_values,
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=current_row, col=1)
        
        fig.add_hline(y=config['rsi_overbought'], line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=config['rsi_oversold'], line_dash="dash", line_color="green", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        
        current_row += 1
    
    # Add MACD if selected
    if 'MACD' in indicators and macd_values:
        start_idx = len(closes) - len(macd_values)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'][start_idx:],
            y=[macd['macd'] for macd in macd_values],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=current_row, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'][start_idx:],
            y=[macd['signal'] for macd in macd_values],
            mode='lines',
            name='Signal',
            line=dict(color='red', width=1)
        ), row=current_row, col=1)
        
        fig.add_trace(go.Bar(
            x=df['timestamp'][start_idx:],
            y=[macd['histogram'] for macd in macd_values],
            name='Histogram',
            marker_color='gray',
            opacity=0.6
        ), row=current_row, col=1)
    
    fig.update_layout(
        title=f'Advanced Technical Analysis - {symbol}',
        xaxis_rangeslider_visible=False,
        height=600 + (200 * (subplot_count - 1)),
        showlegend=True,
        template="plotly_dark"
    )
    
    return fig

def display_technical_summary(symbol: str, klines: List[Dict], config: Dict):
    """Display current technical analysis summary"""
    if not klines:
        return
    
    closes = [k['close'] for k in klines]
    current_rsi = calculate_rsi(closes, config['rsi_period'])
    current_macd = calculate_macd(closes, config['macd_fast'], config['macd_slow'], config['macd_signal'])
    current_ma = calculate_moving_averages(closes, config['ma_short'], config['ma_long'])
    current_bb = calculate_bollinger_bands(closes)
    
    st.subheader("📊 Current Technical Indicators")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        rsi_status = "Oversold" if current_rsi < config['rsi_oversold'] else "Overbought" if current_rsi > config['rsi_overbought'] else "Neutral"
        rsi_color = "🟢" if current_rsi < config['rsi_oversold'] else "🔴" if current_rsi > config['rsi_overbought'] else "🟡"
        
        st.metric(
            label="RSI",
            value=f"{rsi_color} {current_rsi:.1f}",
            delta=rsi_status
        )
    
    with tech_col2:
        macd_trend = "Bullish" if current_macd['macd'] > current_macd['signal'] else "Bearish"
        macd_emoji = "📈" if current_macd['macd'] > current_macd['signal'] else "📉"
        
        st.metric(
            label="MACD",
            value=f"{current_macd['macd']:.6f}",
            delta=f"{macd_emoji} {macd_trend}"
        )
    
    with tech_col3:
        ma_trend = "Bullish" if current_ma['ma_short'] > current_ma['ma_long'] else "Bearish"
        ma_emoji = "📈" if current_ma['ma_short'] > current_ma['ma_long'] else "📉"
        
        st.metric(
            label=f"MA {config['ma_short']}/{config['ma_long']}",
            value=f"${current_ma['ma_short']:,.2f}",
            delta=f"{ma_emoji} {ma_trend}"
        )
    
    with tech_col4:
        current_price = closes[-1]
        if current_bb['upper'] > 0:
            bb_position = "Upper" if current_price > current_bb['upper'] else "Lower" if current_price < current_bb['lower'] else "Middle"
        else:
            bb_position = "N/A"
        
        st.metric(
            label="Bollinger Position",
            value=bb_position,
            delta=f"${current_price:,.4f}"
        )

def display_system_status(config: Dict):
    """Display system status and diagnostics"""
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        # Check Binance API
        try:
            test_response = requests.get(f"{config['binance_api']}/time", timeout=5)
            api_status = "✅ Connected" if test_response.status_code == 200 else "❌ Error"
            api_latency = f"{test_response.elapsed.total_seconds()*1000:.0f}ms"
        except:
            api_status = "❌ Disconnected"
            api_latency = "N/A"
        
        st.metric("Binance API", api_status, delta=api_latency)
    
    with status_col2:
        # Check Telegram status
        if config['telegram_enabled']:
            telegram_status = "✅ Enabled" if config['telegram_token'] and config['telegram_chat_id'] else "⚠️ Incomplete"
        else:
            telegram_status = "⚪ Disabled"
        
        st.metric("Telegram Bot", telegram_status)
    
    with status_col3:
        # Memory usage
        memory_usage = len(str(st.session_state))
        st.metric("Session Memory", f"{memory_usage/1024:.1f} KB")

def export_signals_to_json(signals: List[Signal]):
    """Export signals to JSON format"""
    signals_data = []
    for signal in signals:
        signal_dict = asdict(signal)
        signal_dict['timestamp_formatted'] = datetime.fromtimestamp(signal.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        signals_data.append(signal_dict)
    
    json_str = json.dumps(signals_data, ensure_ascii=False, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Signals JSON",
        data=json_str,
        file_name=f"crypto_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def display_quick_market_overview(config: Dict):
    """Display quick market overview"""
    st.subheader("📊 Quick Market Overview")
    
    market_data = []
    for symbol in config['symbols'][:6]:  # Show top 6 symbols
        ticker = fetch_ticker(symbol)
        if ticker['price'] > 0:
            market_data.append({
                'Symbol': symbol,
                'Price': f"${ticker['price']:,.4f}",
                'Change': f"{ticker['percentage']:+.2f}%",
                'Volume': f"${ticker['volume_24h']/1000000:.1f}M"
            })
    
    if market_data:
        df = pd.DataFrame(market_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

def display_signal_history():
    """Display signal history with filtering"""
    if 'signal_history' not in st.session_state or not st.session_state.signal_history:
        st.info("No signal history available")
        return
    
    history_data = []
    for record in st.session_state.signal_history:
        history_data.append({
            'Symbol': record['symbol'],
            'Side': record['side'],
            'Entry': f"${record.get('entry', 0):,.4f}",
            'Confidence': f"{record.get('confidence', 0):.0f}%",
            'Time': datetime.fromtimestamp(record['timestamp']).strftime("%Y-%m-%d %H:%M:%S"),
            'Signal ID': record.get('signal_id', 'N/A')
        })
    
    df_history = pd.DataFrame(history_data)
    
    # Filter options
    filter_col1, filter_col2 = st.columns([1, 3])
    
    with filter_col1:
        filter_symbol = st.selectbox(
            "Filter by Symbol:",
            ["All"] + list(df_history['Symbol'].unique()),
            index=0
        )
    
    with filter_col2:
        date_range = st.date_input(
            "Date Range:",
            value=[datetime.now().date() - timedelta(days=7), datetime.now().date()],
            format="YYYY-MM-DD"
        )
    
    # Apply filters
    filtered_df = df_history.copy()
    
    if filter_symbol != "All":
        filtered_df = filtered_df[filtered_df['Symbol'] == filter_symbol]
    
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.signal_history = []
        st.success("History cleared!")
        st.rerun()

def display_performance_metrics():
    """Display performance metrics of signals"""
    if 'signal_history' not in st.session_state or not st.session_state.signal_history:
        return
    
    st.subheader("📊 Performance Statistics")
    
    history = st.session_state.signal_history
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Total Signals", len(history))
    
    with stats_col2:
        buy_signals = len([s for s in history if s['side'] == 'BUY'])
        st.metric("Buy Signals", buy_signals)
    
    with stats_col3:
        sell_signals = len([s for s in history if s['side'] == 'SELL'])
        st.metric("Sell Signals", sell_signals)
    
    with stats_col4:
        if history:
            most_active = max(set([s['symbol'] for s in history]), key=[s['symbol'] for s in history].count)
            st.metric("Most Active", most_active)

def display_api_status():
    """Display API connection status"""
    st.subheader("🔗 API Connection Status")
    
    api_col1, api_col2 = st.columns(2)
    
    with api_col1:
        st.write("**Binance API:**")
        try:
            response = requests.get(f"{DEFAULT_CONFIG['binance_api']}/time", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected")
                latency = response.elapsed.total_seconds() * 1000
                st.write(f"Latency: {latency:.0f}ms")
            else:
                st.error(f"❌ Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    with api_col2:
        st.write("**Fear & Greed API:**")
        try:
            response = requests.get(DEFAULT_CONFIG['fear_greed_api'], timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error(f"❌ Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

def display_performance_diagnostics():
    """Display system performance diagnostics"""
    st.subheader("⚡ Performance Diagnostics")
    
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    
    with diag_col1:
        # Session state items count
        session_items = len(st.session_state.keys()) if hasattr(st.session_state, 'keys') else 0
        st.metric("Session Items", session_items)
    
    with diag_col2:
        # Session state size
        session_size = len(str(st.session_state))
        st.metric("Session Size", f"{session_size/1024:.1f} KB")
    
    with diag_col3:
        # Signal history count
        history_count = len(st.session_state.get('signal_history', []))
        st.metric("Signals in History", history_count)
    
    # Additional diagnostics
    diag_col4, diag_col5 = st.columns(2)
    
    with diag_col4:
        # Memory usage estimation
        import sys
        if 'signal_history' in st.session_state:
            history_size = sys.getsizeof(str(st.session_state.signal_history))
            st.metric("History Memory", f"{history_size/1024:.1f} KB")
        else:
            st.metric("History Memory", "0 KB")
    
    with diag_col5:
        # Current time
        current_time = datetime.now()
        st.metric("System Time", current_time.strftime("%H:%M:%S"))
    
    # System recommendations
    if session_size > 100000:  # 100KB
        st.warning("⚠️ Session state is large. Consider clearing history to improve performance.")
    
    if history_count > 100:
        st.info("💡 Consider archiving old signals to maintain optimal performance.")
    
    # Performance actions
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
    
            with chat_col6:
               if st.button("🗑️ Clear All Data"):
                # Clear all session data
                keys_to_remove = list(st.session_state.keys())
                for key in keys_to_remove:
                    del st.session_state[key]
                st.success("All session data cleared!")
                st.rerun()

def display_inducement_levels(symbol: str, inducement_data: Dict):
    """Display inducement levels analysis"""
    st.subheader(f"🎯 Inducement Levels for {symbol}")
    
    current_price = inducement_data.get('current_price', 0)
    resistance_levels = inducement_data.get('resistance_levels', [])
    support_levels = inducement_data.get('support_levels', [])
    
    # Overview metrics
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        st.metric("Current Price", f"${current_price:.4f}")
    
    with overview_col2:
        st.metric("Resistance Levels", len(resistance_levels))
    
    with overview_col3:
        st.metric("Support Levels", len(support_levels))
    
    with overview_col4:
        total_levels = len(resistance_levels) + len(support_levels)
        st.metric("Total Active Levels", total_levels)
    
    # Display levels in columns
    levels_col1, levels_col2 = st.columns(2)
    
    with levels_col1:
        st.write("**🔴 Resistance Levels (Above Price)**")
        if resistance_levels:
            resistance_data = []
            for i, level in enumerate(resistance_levels[:10]):  # Show top 10
                distance = ((level.price - current_price) / current_price) * 100
                strength = "Strong" if distance < 2 else "Moderate" if distance < 5 else "Weak"
                
                resistance_data.append({
                    'Level': f"${level.price:.4f}",
                    'Distance': f"{distance:.2f}%",
                    'Strength': strength,
                    'Age': f"{(time.time() - level.created_at)/3600:.1f}h"
                })
            
            df_resistance = pd.DataFrame(resistance_data)
            
            # Color code based on distance
            def color_resistance(row):
                if 'Strong' in str(row['Strength']):
                    return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                elif 'Moderate' in str(row['Strength']):
                    return ['background-color: rgba(255, 165, 0, 0.1)'] * len(row)
                else:
                    return ['background-color: rgba(128, 128, 128, 0.1)'] * len(row)
            
            styled_resistance = df_resistance.style.apply(color_resistance, axis=1)
            st.dataframe(styled_resistance, hide_index=True, use_container_width=True)
        else:
            st.info("No resistance levels found")
    
    with levels_col2:
        st.write("**🟢 Support Levels (Below Price)**")
        if support_levels:
            support_data = []
            for i, level in enumerate(support_levels[:10]):  # Show top 10
                distance = ((current_price - level.price) / current_price) * 100
                strength = "Strong" if distance < 2 else "Moderate" if distance < 5 else "Weak"
                
                support_data.append({
                    'Level': f"${level.price:.4f}",
                    'Distance': f"{distance:.2f}%",
                    'Strength': strength,
                    'Age': f"{(time.time() - level.created_at)/3600:.1f}h"
                })
            
            df_support = pd.DataFrame(support_data)
            
            # Color code based on distance
            def color_support(row):
                if 'Strong' in str(row['Strength']):
                    return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                elif 'Moderate' in str(row['Strength']):
                    return ['background-color: rgba(255, 165, 0, 0.1)'] * len(row)
                else:
                    return ['background-color: rgba(128, 128, 128, 0.1)'] * len(row)
            
            styled_support = df_support.style.apply(color_support, axis=1)
            st.dataframe(styled_support, hide_index=True, use_container_width=True)
        else:
            st.info("No support levels found")
    
    # Key levels summary
    st.subheader("📊 Key Levels Analysis")
    
    # Find closest levels
    closest_resistance = None
    closest_support = None
    
    for level in resistance_levels:
        if level.price > current_price:
            if not closest_resistance or level.price < closest_resistance.price:
                closest_resistance = level
    
    for level in support_levels:
        if level.price < current_price:
            if not closest_support or level.price > closest_support.price:
                closest_support = level
    
    key_col1, key_col2 = st.columns(2)
    
    with key_col1:
        if closest_resistance:
            resistance_distance = ((closest_resistance.price - current_price) / current_price) * 100
            st.info(f"""
            **🔴 Nearest Resistance:**
            - Price: ${closest_resistance.price:.4f}
            - Distance: {resistance_distance:.2f}%
            - Risk Level: {'HIGH' if resistance_distance < 1 else 'MEDIUM' if resistance_distance < 3 else 'LOW'}
            """)
        else:
            st.info("No resistance levels above current price")
    
    with key_col2:
        if closest_support:
            support_distance = ((current_price - closest_support.price) / current_price) * 100
            st.info(f"""
            **🟢 Nearest Support:**
            - Price: ${closest_support.price:.4f}
            - Distance: {support_distance:.2f}%
            - Risk Level: {'HIGH' if support_distance < 1 else 'MEDIUM' if support_distance < 3 else 'LOW'}
            """)
        else:
            st.info("No support levels below current price")
    
    # Trading recommendations based on inducement levels
    st.subheader("💡 Trading Insights")
    
    recommendations = []
    
    if closest_resistance and closest_support:
        resistance_dist = ((closest_resistance.price - current_price) / current_price) * 100
        support_dist = ((current_price - closest_support.price) / current_price) * 100
        
        if resistance_dist < 1:
            recommendations.append("⚠️ Price approaching strong resistance - Consider taking profits")
        
        if support_dist < 1:
            recommendations.append("⚠️ Price approaching strong support - Potential buying opportunity")
        
        if resistance_dist > 3 and support_dist > 3:
            recommendations.append("📈 Price in neutral zone - Wait for clear direction")
        
        # Risk/Reward analysis
        rr_ratio = resistance_dist / support_dist if support_dist > 0 else 0
        if rr_ratio > 2:
            recommendations.append("📊 Favorable R/R ratio for long positions")
        elif rr_ratio < 0.5:
            recommendations.append("📊 Favorable R/R ratio for short positions")
    
    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.write("• No specific recommendations at this time")

if __name__ == "__main__":
    main()