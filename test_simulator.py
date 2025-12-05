#!/usr/bin/env python3
"""
Test Simulator for TradeGPT Scalper
Simulates 20 trades in test mode to evaluate performance
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import random
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import bot components (mocked for simulation)
from tradegpt_scalper import TradeGPTScalper, Position, EFFECTIVE_EXPOSURE, MIN_PROFIT_USD, MAX_LOSS_USD, TRAILING_ACTIVATION_USD, TRAILING_DISTANCE_PCT

# Mock Bybit API for simulation
class MockBybitAPI:
    def __init__(self):
        self.positions = []
        self.prices = {}
        self.trades_history = []
        logger.info("Mock Bybit API initialized for simulation")
    
    async def fetch_ticker(self, symbol):
        # Simulate price with random walk
        if symbol not in self.prices:
            self.prices[symbol] = 100.0
        
        # Simulate price movement (-1% to +1% change)
        change = random.uniform(-0.01, 0.01)
        self.prices[symbol] *= (1 + change)
        
        return {'last': self.prices[symbol], 'bid': self.prices[symbol] * 0.999, 'ask': self.prices[symbol] * 1.001}
    
    async def fetch_ohlcv(self, symbol, timeframe, limit=20):
        # Simulate historical candles
        if symbol not in self.prices:
            self.prices[symbol] = 100.0
        
        candles = []
        current_price = self.prices[symbol]
        for i in range(limit):
            change = random.uniform(-0.01, 0.01)
            prev_price = current_price
            current_price *= (1 + change)
            high = max(prev_price, current_price) * 1.002
            low = min(prev_price, current_price) * 0.998
            volume = random.uniform(100, 1000)
            candles.append([0, prev_price, high, low, current_price, volume])
        return candles[::-1]  # Reverse to simulate chronological order
    
    async def fetch_positions(self):
        return self.positions
    
    async def create_order(self, symbol, side, amount, price=None, leverage=20):
        entry_price = self.prices[symbol]
        order_id = f"sim_order_{len(self.trades_history)}"
        position = {
            'symbol': symbol,
            'side': 'Buy' if side == 'buy' else 'Sell',
            'entryPrice': entry_price,
            'contracts': amount,
            'stopLoss': 0.0,
            'takeProfit': 0.0,
            'unrealisedPnl': 0.0,
            'id': order_id
        }
        self.positions.append(position)
        self.trades_history.append({
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'amount': amount,
            'timestamp': time.time(),
            'status': 'open'
        })
        logger.info(f"Simulated order created: {symbol} {side} @ {entry_price}")
        return {'id': order_id}
    
    async def set_stop_loss(self, symbol, side, sl_price):
        for pos in self.positions:
            if pos['symbol'] == symbol and pos['side'] == ('Buy' if side == 'long' else 'Sell'):
                pos['stopLoss'] = sl_price
                logger.info(f"Simulated SL set: {symbol} @ {sl_price}")
                return True
        return False
    
    async def set_take_profit(self, symbol, side, tp_price):
        for pos in self.positions:
            if pos['symbol'] == symbol and pos['side'] == ('Buy' if side == 'long' else 'Sell'):
                pos['takeProfit'] = tp_price
                logger.info(f"Simulated TP set: {symbol} @ {tp_price}")
                return True
        return False
    
    async def close_position(self, symbol):
        for i, pos in enumerate(self.positions):
            if pos['symbol'] == symbol:
                current_price = self.prices[symbol]
                entry_price = pos['entryPrice']
                side = pos['side']
                amount = pos['contracts']
                pnl = (current_price - entry_price) * amount if side == 'Buy' else (entry_price - current_price) * amount
                
                # Update trade history
                for trade in self.trades_history:
                    if trade['symbol'] == symbol and trade['status'] == 'open':
                        trade['exit_price'] = current_price
                        trade['pnl'] = pnl
                        trade['status'] = 'closed'
                        trade['close_timestamp'] = time.time()
                        break
                
                del self.positions[i]
                logger.info(f"Simulated position closed: {symbol} @ {current_price}, PnL: {pnl}")
                return True
        return False
    
    async def get_account_balance(self):
        return {'total': {'USDT': 10000}, 'free': {'USDT': 10000}, 'used': {'USDT': 0}}
    
    async def close(self):
        pass

# Mock Disco57 for simulation
class MockDisco57PPO:
    def predict(self, price, volume_ratio, momentum, volatility):
        # Simulate ALLOW 70% of the time for testing
        return 'ALLOW' if random.random() < 0.7 else 'BLOCK'

# Mock Telegram Notifier for simulation
class MockTelegramNotifier:
    async def send_message(self, message, parse_mode='HTML'):
        logger.info(f"Simulated Telegram message: {message}")
        return True

# Simulator class
class TradeGPTSimulator(TradeGPTScalper):
    def __init__(self, max_trades=20):
        self.api = MockBybitAPI()
        self.disco57 = MockDisco57PPO()
        self.telegram = MockTelegramNotifier()
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.symbol_cooldowns: Dict[str, float] = {}
        self.max_trades = max_trades
        self.closed_trades = []
        
        logger.info(f"TradeGPT Simulator initialized for {max_trades} trades")
    
    async def start(self):
        logger.info("=" * 60)
        logger.info(f"TradeGPT Simulator started for {self.max_trades} trades")
        logger.info("=" * 60)
        
        # Load active positions (simulated)
        await self.load_active_positions()
        
        # Main loop until max trades reached
        while len(self.closed_trades) < self.max_trades:
            try:
                await self.main_loop()
                await asyncio.sleep(1)  # Simulate faster cycles for testing
                
                # Check if we need to close positions to reach trade count
                if len(self.closed_trades) < self.max_trades and len(self.positions) > 0:
                    for symbol in list(self.positions.keys()):
                        if len(self.closed_trades) < self.max_trades:
                            ticker = await self.api.fetch_ticker(symbol)
                            current_price = float(ticker['last'])
                            await self.close_position(symbol, current_price, "SIMULATION")
            except KeyboardInterrupt:
                logger.info("Simulation stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        # Summarize results
        self.summarize_results()
    
    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close position and record trade for statistics"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        try:
            # Close via API
            await self.api.close_position(symbol)
            
            # Wait for confirmation
            start_time = time.time()
            from tradegpt_scalper import POSITION_CLOSE_MAX_WAIT, POSITION_CLOSE_CHECK_INTERVAL
            while time.time() - start_time < POSITION_CLOSE_MAX_WAIT:
                positions = await self.api.fetch_positions()
                if not any(p['symbol'] == symbol and float(p.get('contracts', 0)) > 0 for p in positions):
                    logger.info(f"Confirmed position closed {symbol} via API")
                    break
                logger.debug(f"Waiting for position close {symbol}...")
                await asyncio.sleep(POSITION_CLOSE_CHECK_INTERVAL)
            else:
                logger.warning(f"Timeout waiting for {symbol} to close, possible phantom position")
            
            # Calculate PnL
            if pos.side == 'long':
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            
            pnl_usd = pnl_pct * EFFECTIVE_EXPOSURE
            self.daily_pnl += pnl_usd
            
            # Record trade
            self.closed_trades.append({
                'symbol': symbol,
                'side': pos.side,
                'entry_price': pos.entry_price,
                'exit_price': exit_price,
                'pnl_usd': pnl_usd,
                'reason': reason,
                'entry_time': pos.entry_time,
                'exit_time': time.time()
            })
            
            logger.info(f"❌ CLOSED POSITION: {symbol} | {reason} | PnL: ${pnl_usd:.2f} | Trade {len(self.closed_trades)}/{self.max_trades}")
            
            # Telegram notification (simulated)
            status_emoji = "✅" if pnl_usd > 0 else "❌"
            await self.telegram.send_message(
                f"{status_emoji} CLOSED ({reason})\n"
                f"{symbol} {pos.side.upper()}\n"
                f"Entry: ${pos.entry_price:.6f}\n"
                f"Exit: ${exit_price:.6f}\n"
                f"PnL: ${pnl_usd:+.2f}\n"
                f"Daily PnL: ${self.daily_pnl:+.2f}"
            )
            
            # Remove from positions
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
    
    def summarize_results(self):
        """Summarize simulation results"""
        logger.info("=" * 60)
        logger.info(f"Simulation completed: {len(self.closed_trades)} trades executed")
        logger.info("=" * 60)
        
        total_pnl = sum(trade['pnl_usd'] for trade in self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade['pnl_usd'] > 0)
        win_rate = (winning_trades / len(self.closed_trades)) * 100 if self.closed_trades else 0
        avg_pnl = total_pnl / len(self.closed_trades) if self.closed_trades else 0
        max_win = max((trade['pnl_usd'] for trade in self.closed_trades), default=0)
        max_loss = min((trade['pnl_usd'] for trade in self.closed_trades), default=0)
        
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.1f}% ({winning_trades}/{len(self.closed_trades)})")
        logger.info(f"Average PnL per Trade: ${avg_pnl:.2f}")
        logger.info(f"Max Win: ${max_win:.2f}")
        logger.info(f"Max Loss: ${max_loss:.2f}")
        
        # Detailed trade log
        logger.info("\nDetailed Trade Log:")
        for i, trade in enumerate(self.closed_trades, 1):
            logger.info(f"Trade {i}: {trade['symbol']} {trade['side'].upper()} | Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | PnL: ${trade['pnl_usd']:.2f} | Reason: {trade['reason']}")
        
        logger.info("=" * 60)
        logger.info("End of Simulation Report")
        logger.info("=" * 60)


# Main simulation function
async def run_simulation(max_trades=20):
    """Run simulation for specified number of trades"""
    simulator = TradeGPTSimulator(max_trades=max_trades)
    await simulator.start()


if __name__ == '__main__':
    asyncio.run(run_simulation(max_trades=20))
