from abc import ABC
from logging import getLogger, INFO

from krakenex import API

from trade.private import KEY, SECRET

logger = getLogger(__name__)
logger.setLevel(INFO)


class BaseTrader(ABC):
    def __init__(self):
        logger.debug("Initializing Kraken API...")
        self.client = API(key=KEY, secret=SECRET)

    def get_ticker(self, pair: str):
        r = self.client.query_public('Ticker', data=dict(pair=pair))
        return self._handle_request(r)

    @staticmethod
    def mid_price(ticker) -> float:
        return round(0.5 * (float(ticker['a'][0]) + float(ticker['b'][0])), 4)

    @staticmethod
    def _latest_close_price(ticker) -> float:
        return round(float(ticker['c'][0]), 4)

    def query_order(self, txid: str, **kwargs):
        r = self.client.query_private('QueryOrders', data={**dict(txid=txid), **kwargs})
        return self._handle_request(r)

    def query_trade(self, txid: str, **kwargs):
        r = self.client.query_private('QueryTrades', data={**dict(txid=txid), **kwargs})
        return self._handle_request(r)

    @staticmethod
    def _handle_request(r):
        if r['error']:
            logger.error(f"[API] Error handling request: {r['error']}")
            raise Exception(f"[API] Error handling request: {r['error']}")
        return r['result']


class PairsTrader(BaseTrader):
    def __init__(self, pair_a: str, pair_b: str):
        super(PairsTrader, self).__init__()
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.id = abs(hash(f"{pair_a}{pair_b}")) % (10 ** 6)

    def _ez_limit_order(self, order_type: str, pair: str, userref: int, limit_price: float, volume: float, fos: float,
                        leverage: int = None):
        assert (0 < fos < 1, "Factor of safety must be between 0-1")
        fos = (1 + fos) if order_type == 'buy' else (1 - fos)
        r = self.client.query_private('AddOrder', data=dict(
            userref=userref,
            ordertype='limit',
            type=order_type,
            volume=round(volume, 8),
            pair=pair,
            price=round(fos * limit_price, 4),
            timeinforce="IOC",
            leverage=leverage,
        ))
        r = self._handle_request(r)

        logger.debug(f"[Trader] {r['descr']['order']}")

        orders = []
        for o_id in r['txid']:
            o = self.query_order(txid=o_id, userref=userref, trade=True)
            orders.append(o)
            status = o[o_id]['status']
            assert (status == 'closed', f"[Trader] Order ({o_id}) not closed, status: {status}.")

        return r, orders

    def _profits_from_orders(self, *orders):
        trades = []
        for o in orders:
            for _, v in o.items():
                trades.extend(v['trades'])
        # Get opening trades
        trades = [self.query_trade(txid=t_id, trades=True)[t_id]['postxid'] for t_id in trades]
        cost, net = 0, 0
        for t_id in trades:
            t = self.query_trade(txid=t_id, trades=True)
            cost += float(t[t_id]['ccost'])
            net += float(t[t_id]['net'])

        return net, net / cost

    def go_long(self, counter: int, volume_a: float, volume_b: float, price_a: float, price_b: float, fos: float):
        ref = int(f"{self.id}{counter}")
        try:
            r_a, o_a = self._ez_limit_order(order_type='sell', pair=self.pair_a, userref=ref, limit_price=price_a,
                                            volume=volume_a, fos=fos, leverage=2)
            r_b, o_b = self._ez_limit_order(order_type='buy', pair=self.pair_b, userref=ref, limit_price=price_b,
                                            volume=volume_b, fos=fos, leverage=2)
        except Exception as e:
            logger.error(f"[Trader] Failed to open long positions: {e}")
            raise e

        return r_a, r_b, o_a, o_b

    def go_short(self, counter: int, volume_a: float, volume_b: float, price_a: float, price_b: float, fos: float):
        ref = int(f"{self.id}{counter}")
        try:
            r_a, o_a = self._ez_limit_order(order_type='buy', pair=self.pair_a, userref=ref, limit_price=price_a,
                                            volume=volume_a, fos=fos, leverage=2)
            r_b, o_b = self._ez_limit_order(order_type='sell', pair=self.pair_b, userref=ref, limit_price=price_b,
                                            volume=volume_b, fos=fos, leverage=2)

        except Exception as e:
            logger.error(f"[Trader] Failed to open short positions: {e}")
            raise e

        return r_a, r_b, o_a, o_b

    def close_long(self, counter: int, price_a: float, price_b: float, fos: float):
        try:
            _, _, o_a, o_b = self.go_short(counter, 0, 0, price_a, price_b, fos)
            return self._profits_from_orders(*o_a, *o_b)
        except Exception as e:
            logger.error(f"[Trader] Failed to close long positions: {e}")
            raise e

    def close_short(self, counter: int, price_a: float, price_b: float, fos: float):
        try:
            _, _, o_a, o_b = self.go_long(counter, 0, 0, price_a, price_b, fos)
            return self._profits_from_orders(*o_a, *o_b)
        except Exception as e:
            logger.error(f"[Trader] Failed to close short positions: {e}")
            raise e
