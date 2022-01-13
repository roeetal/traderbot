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

    def __init__(self, pair_a: str, pair_b: str, path: str, bake_count: int, volume: float, th: float, sleep_time: int,
                 save_time: int):
        super(Pairs, self).__init__(path=os.path.join(path, f'{pair_a}_{pair_b}_strategy'))
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.model = KalmanOLS(path=os.path.join(path, f'{pair_a}_{pair_b}_model'))
        self.bake_count = bake_count
        self.trader = PairsTrader(pair_a, pair_b)
        self.th = th
        self.sleep_time = sleep_time
        self.save_time = save_time
        self.volume_b = round(volume, 8)
        try:
            state = self.load_state()
            self.counter = state['counter']
            self.position = state['invested']
        except FileNotFoundError:
            logger.debug("Initialising state with defaults...")
            self.counter = 0
            self.position = Position.NOT_INVESTED
        logger.info("Strategy initialized!")

    def get_state(self) -> Dict[str, Any]:
        return {
            "counter": self.counter,
            "invested": self.position,
        }

    def __call__(self):
        self.counter += 1
        sleep_time = self.sleep_time

        ticker_a, ticker_b = (self.trader.get_ticker(x)[x] for x in (self.pair_a, self.pair_b))
        price_a, price_b = (self.trader.mid_price(ticker) for ticker in (ticker_a, ticker_b))
        logger.debug(f"{self.pair_a}: {price_a:.2f}, {self.pair_b}: {price_b:.2f}")
        e, q, t = self.model(price_a, price_b)
        logger.debug(f"error: ${e:.4f}, var: $${q:.4f}, b_weight:{t:.4f}")

        if self.counter > self.bake_count:
            old_position = self.position

            if self.position == Position.NOT_INVESTED:
                if e < - np.sqrt(q):
                    logger.info(f"Going long: ${e:.2f} < - ${np.sqrt(q):.2f}")
                    volume_a = round(self.volume_b * t, 8)
                    self.trader.go_long(self.counter, volume_a, self.volume_b, price_a, price_b, self.th)
                    self.position = Position.LONG
                elif e > np.sqrt(q):
                    logger.info(f"Going short: {e:.2f} > {np.sqrt(q):.2f}")
                    volume_a = round(self.volume_b * t, 8)
                    self.trader.go_short(self.counter, volume_a, self.volume_b, price_a, price_b, self.th)
                    self.position = Position.SHORT
                else:
                    logger.debug(f"Abstaining")
            elif self.position == Position.LONG:
                if e > - np.sqrt(q):
                    logger.info(f"Closing long: {e:.2f} > - {np.sqrt(q):.2f}")
                    p, r = self.trader.close_long(self.counter, price_a, price_b, self.th)
                    logger.info(f"Profit: ${p:.2f}")
                    logger.info(f"Return: {r * 100:.2f}%")
                    self.position = Position.NOT_INVESTED
                else:
                    logger.debug(f"Staying long")
            elif self.position == Position.SHORT:
                if e < np.sqrt(q):
                    logger.info(f"Closing short: {e:.2f} < {np.sqrt(q):.2f}")
                    p, r = self.trader.close_short(self.counter, price_a, price_b, self.th)
                    logger.info(f"Profit: ${p:.2f}")
                    logger.info(f"Return: {r * 100:.2f}%")
                    self.position = Position.NOT_INVESTED
                else:
                    logger.debug(f"Staying short")

            if old_position != self.position or self.counter % (self.save_time / self.sleep_time) == 0:
                self.model.save_state()
        else:
            logger.debug(f"Incrementing bake count: {self.counter}")
            self.model.save_state()
            sleep_time = 4 * 60 * 60

        self.save_state()

        logger.debug(f"Returning sleep time: {sleep_time} seconds")
        return sleep_time


if __name__ == '__main__':
    signal(SIGINT, signal_handler)

    parser = ArgumentParser()
    parser.add_argument("-a", "--pair-a", help="pair a", type=str, required=True)
    parser.add_argument("-b", "--pair-b", help="pair b", type=str, required=True)
    parser.add_argument("-p", "--path", help="path", type=str, required=True)
    parser.add_argument("-bc", "--bake-count", help="bake count", type=int, required=False, default=2)
    parser.add_argument("-v", "--volume", help="volume of pair b", type=float, required=False, default=1.0)
    parser.add_argument("-th", "--threshold", help="limit order threshold", type=float, required=False, default=0.01)
    parser.add_argument("-st", "--sleep-time", help="sleep time in seconds", type=int, required=False, default=30)
    parser.add_argument("-sc", "--save-time", help="max time before save", type=int, required=False, default=46800)
    args = parser.parse_args()

    strategy = Pairs(args.pair_a, args.pair_b, args.path, args.bake_count, args.volume, args.threshold, args.sleep_time,
                     args.save_time)
    while True:
        st = strategy()
        sleep(st)
