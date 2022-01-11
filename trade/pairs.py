import os
import sys
from argparse import ArgumentParser
from enum import Enum
from logging import getLogger, INFO, basicConfig, DEBUG, Formatter, CRITICAL, ERROR
from signal import signal, SIGINT
from time import sleep
from typing import Dict, Any

import numpy as np

from trade.defaults import Stateful
from trade.logger import SlackHandler
from trade.models import KalmanOLS
from trade.private import SLACK_TOKEN
from trade.traders import PairsTrader

logging_format = '%(asctime)s  [%(name)s] (%(levelname)s): %(message)s'
basicConfig(filename='logs/pairs_logs', level=DEBUG, format=logging_format)
logger = getLogger("statistical.arbitrage")
formatter = Formatter(logging_format)

for level in ("debug", "info", "error"):
    sh = SlackHandler(username='logger', token=SLACK_TOKEN, channel=f'#{level}', fmt=formatter)
    sh.setLevel(INFO if level == "info" else ERROR if level == "error" else CRITICAL if level == "critical" else DEBUG)
    logger.addHandler(sh)


def signal_handler(sig, frame):
    logger.error(f"Shutting down... ({sig})")
    sys.exit(0)


class Position(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NOT_INVESTED = "NOT_INVESTED"


class Pairs(Stateful):

    def __init__(self, pair_a: str, pair_b: str, path: str, bake_count: int, volume: float, th: float):
        super(Pairs, self).__init__(path=os.path.join(path, f'{pair_a}_{pair_b}_strategy'))
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.model = KalmanOLS(path=os.path.join(path, f'{pair_a}_{pair_b}_model'))
        self.bake_count = bake_count
        self.trader = PairsTrader(pair_a, pair_b)
        self.th = th
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
        logger.info("Strategy initialized!")

    def get_state(self) -> Dict[str, Any]:
        return {
            "counter": self.counter,
            "invested": self.position,
            "volume_a": self.volume_a,
            "volume_b": self.volume_b,
        }

    def __call__(self):
        self.counter += 1
        ticker_a, ticker_b = (self.trader.get_ticker(x)[x] for x in (self.pair_a, self.pair_b))
        price_a, price_b = (self.trader.mid_price(ticker) for ticker in (ticker_a, ticker_b))
        logger.debug(f"[Strategy] {self.pair_a}: {price_a:.2f}, {self.pair_b}: {price_b:.2f}")
        e, q, t = self.model(price_a, price_b)
        logger.debug(f"[Strategy] error: ${e:.4f}, var: $${q:.4f}, b_weight:{t:.4f}")

        if self.counter > self.bake_count:
            if self.position == Position.NOT_INVESTED:
                if e < - 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Going long: ${e:.2f} < - ${2 * np.sqrt(q):.2f}")
                    self.volume_a = round(self.volume_b * t, 8)
                    self.trader.go_long(self.counter, self.volume_a, self.volume_b, price_a, price_b, self.th)
                    self.position = Position.LONG
                elif e > 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Going short: {e} > {2 * np.sqrt(q)}")
                    self.volume_a = round(self.volume_b * t, 8)
                    self.trader.go_short(self.counter, self.volume_a, self.volume_b, price_a, price_b, self.th)
                    self.position = Position.SHORT
                else:
                    logger.debug(f"[Strategy] Abstaining")
            elif self.position == Position.LONG:
                if e > - 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Closing long: {e} > - {2 * np.sqrt(q)}")
                    p, r = self.trader.close_long(self.counter, price_a, price_b, self.th)
                    logger.info(f"[Strategy] Profit: ${p:.2f}")
                    logger.info(f"[Strategy] Return: {r * 100:.2f}%")
                    self.position = Position.NOT_INVESTED
                else:
                    logger.debug(f"[Strategy] Staying long")
            elif self.position == Position.SHORT:
                if e < 2 * np.sqrt(q):
                    logger.info(f"[Strategy] Closing short: {e} < {2 * np.sqrt(q)}")
                    p, r = self.trader.close_short(self.counter, price_a, price_b, self.th)
                    logger.info(f"[Strategy] Profit: ${p:.2f}")
                    logger.info(f"[Strategy] Return: {r * 100:.2f}%")
                    self.position = Position.NOT_INVESTED
                else:
                    logger.debug(f"[Strategy] Staying short")
        else:
            logger.debug(f"[Strategy] Incrementing bake count: {self.counter}")

        self.save_state()


if __name__ == '__main__':
    signal(SIGINT, signal_handler)

    parser = ArgumentParser()
    parser.add_argument("-a", "--pair-a", help="pair a", type=str, required=True)
    parser.add_argument("-b", "--pair-b", help="pair b", type=str, required=True)
    parser.add_argument("-p", "--path", help="path", type=str, required=True)
    parser.add_argument("-bc", "--bake-count", help="bake count", type=int, required=False, default=2)
    parser.add_argument("-v", "--volume", help="volume of pair b", type=float, required=False, default=1.0)
    parser.add_argument("-th", "--threshold", help="limit order threshold", type=float, required=False, default=0.01)
    parser.add_argument("-h", "--hours", help="interval", type=int, required=False, default=4)
    args = parser.parse_args()

    strategy = Pairs(args.pair_a, args.pair_b, args.path, args.bake_count, args.volume, args.threshold)
    while True:
        strategy()
        sleep(60*60*args.hours)
