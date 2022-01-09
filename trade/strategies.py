import os
import sys
from argparse import ArgumentParser
from logging import getLogger, INFO, basicConfig
from typing import Dict, Any
from enum import Enum

import numpy as np

from trade.defaults import Stateful
from trade.models import KalmanOLS
from trade.traders import PairsTrader

basicConfig(stream=sys.stdout, level=INFO)
logger = getLogger(__name__)


class Position(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NOT_INVESTED = "NOT_INVESTED"


class Pairs(Stateful):

    def __init__(self, pair_a: str, pair_b: str, path: str, bake_count: int, volume: float, fos: float):
        super(Pairs, self).__init__(path=os.path.join(path, 'strategy'))
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.model = KalmanOLS(path=os.path.join(path, 'model'))
        self.bake_count = bake_count
        self.trader = PairsTrader(pair_a, pair_b)
        self.fos = fos
        try:
            state = self.load_state()
            self.counter = state['counter']
            self.position = state['invested']
            self.volume_a = state['volume_a']
            self.volume_b = state['volume_b']
        except FileNotFoundError:
            logger.debug("Initialising state with defaults...")
            self.counter = 0
            self.position = Position.NOT_INVESTED
            self.volume_a = round(volume, 8)
            self.volume_b = round(volume, 8)

    def get_state(self) -> Dict[str, Any]:
        return {
            "counter": self.counter,
            "invested": self.position,
            "volume_a": self.volume_a,
            "volume_b": self.volume_b,
        }

    def __call__(self):
        self.counter += 1
        ticker_a, ticker_b = (self.trader.get_ticker(x) for x in (self.pair_a, self.pair_b))
        price_a, price_b = (self.trader.mid_price(ticker) for ticker in (ticker_a, ticker_b))
        logger.debug(f"[Strategy] {self.pair_a}: {price_a:.2f}, {self.pair_b}: {price_b:.2f}")
        e, q, t = self.model(price_a, price_b)
        logger.info(f"[Strategy] error: {e:.4f}$, var: {q:.4f}$, b_weight:{t:.4f}")

        if self.counter > self.bake_count:
            if self.position == Position.NOT_INVESTED:
                if e < - 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Going long: {e} < - {2 * np.sqrt(q)}")
                    self.volume_a = round(self.volume_b * t, 8)
                    self.trader.go_long(self.counter, self.volume_a, self.volume_b, price_a, price_b, self.fos)
                    self.position = Position.LONG
                elif e > 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Going short: {e} > {2 * np.sqrt(q)}")
                    self.volume_a = round(self.volume_b * t, 8)
                    self.trader.go_short(self.counter, self.volume_a, self.volume_b, price_a, price_b, self.fos)
                    self.position = Position.SHORT
                else:
                    logger.debug(f"[Strategy] Abstaining.")
            elif self.position == Position.LONG:
                if e > - 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Closing long: {e} > - {2 * np.sqrt(q)}")
                    p, r = self.trader.close_long(self.counter, price_a, price_b, self.fos)
                    logger.info(f"[Strategy] Profit: {p:.2f}$.")
                    logger.info(f"[Strategy] Return: {r * 100:.2f}%.")
                    self.position = Position.NOT_INVESTED
                else:
                    logger.debug(f"[Strategy] Staying long.")
            elif self.position == Position.SHORT:
                if e < 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Closing short: {e} < {2 * np.sqrt(q)}")
                    p, r = self.trader.close_short(self.counter, price_a, price_b, self.fos)
                    logger.info(f"[Strategy] Profit: {p:.2f}$.")
                    logger.info(f"[Strategy] Return: {r * 100:.2f}%.")
                    self.position = Position.NOT_INVESTED
                else:
                    logger.debug(f"[Strategy] Staying short.")
        else:
            logger.debug(f"[Strategy] Incrementing bake count: {self.counter}")

        self.save_state()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-a", "--pair-a", help="pair a", type=str, required=True)
    parser.add_argument("-b", "--pair-b", help="pair b", type=str, required=True)
    parser.add_argument("-p", "--path", help="path", type=str, required=True)
    parser.add_argument("-bc", "--bake-count", help="bake count", type=int, required=False, default=2)
    parser.add_argument("-v", "--volume", help="volume of pair b", type=float, required=False, default=1.0)
    args = parser.parse_args()

    strategy = Pairs(args.pair_a, args.pair_b, args.path, args.bake_count, args.volume)
    strategy()
