import logging
from decimal import Decimal
from typing import List, Dict

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase

import pandas_ta as ta


class RSIAMM(DirectionalStrategyBase):
    """
    BotCamp Cohort: Sept 2022
    Design Template: https://hummingbot-foundation.notion.site/Simple-PMM-63cc765486dd42228d3da0b32537fc92
    Video: -
    Description:
    The bot will place two orders around the price_source (mid price or last traded price) in a trading_pair on
    exchange, with a distance defined by the ask_spread and bid_spread. Every order_refresh_time in seconds,
    the bot will cancel and replace the orders.
    """
    bid_spread = 1  # NATR
    ask_spread = 1  # NATR
    rsi_refresh_time = 15
    order_amount = 0.0005
    create_timestamp = 0
    trading_pair = "BTC-USDT"
    exchange = "binance"
    interval = "3m"
    natr_period = 100
    TICK = 1
    rsi_signal = 0
    buffer_size = 100
    rsi_spread_change_percentage = 0.5
    # from 0 to 1; if 1 then when RSI shows buy/sell signal then the new mid will be moved to the buy/sell price, and a new proposal based on that

    # Here you can use for example the LastTrade price to use in your strategy (good for illiquid markets)
    price_source = PriceType.MidPrice

    candles = [CandlesFactory.get_candle(
        CandlesConfig(connector=exchange, trading_pair=trading_pair, interval="3m", max_records=1000))]
    markets = {exchange: {trading_pair}}

    last_mid_price = None

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.market_data_provider = MarketDataProvider(connectors)
        self.max_records = self.natr_period + 10
        self.market_data_provider.initialize_candles_feed(
            config=CandlesConfig(connector=self.exchange,
                                 trading_pair=self.trading_pair,
                                 interval=self.interval,
                                 max_records=self.max_records)
        )

    def on_tick(self):
        if not self.market_data_provider.ready:
            return

        new_mid_price = self.market_data_provider.get_price_by_type(self.exchange, self.trading_pair, self.price_source)

        if self.create_timestamp <= self.current_timestamp:
            # calculate RSI, if signal is 1 then BUY signal, -1 is SELL signal. Shift the MID price accordingly
            self.rsi_signal = self.get_rsi_signal()
            self.create_timestamp = self.rsi_refresh_time + self.current_timestamp

        if self.last_mid_price is None or abs(new_mid_price - self.last_mid_price) > self.buffer_size:
            # cancel, propose, and place orders
            msg = (f"Mid price is {new_mid_price}")
            self.log_with_clock(logging.INFO, msg)
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal(new_mid_price)
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.last_mid_price = new_mid_price

    def get_rsi_signal(self):
        """
        Generates the trading signal based on the RSI indicator.
        Returns:
            int: The trading signal (-1 for sell, 0 for hold, 1 for buy).
        """
        candles_df = self.get_processed_df()
        rsi_value = candles_df.iat[-1, -1]
        msg = (f"RSI signal is {rsi_value}")
        self.log_with_clock(logging.INFO, msg)
        if rsi_value > 70:
            self.log_with_clock(logging.INFO, "RSI signal is over 70 - adjusting mid price")
            return -1
        elif rsi_value < 30:
            self.log_with_clock(logging.INFO, "RSI signal is under 30 - adjusting mid price")
            return 1
        else:
            return 0

    def get_processed_df(self):
        """
        Retrieves the processed dataframe with RSI values.
        Returns:
            pd.DataFrame: The processed dataframe with RSI values.
        """
        candles_df = self.candles[0].candles_df
        candles_df.ta.rsi(length=7, append=True)
        return candles_df

    def create_proposal(self, mid_price) -> List[OrderCandidate]:
        natr = self.get_natr()
        if self.rsi_signal == 1:
            adjusted_mid_price = mid_price * Decimal(1 + (self.bid_spread * natr * self.rsi_spread_change_percentage))
        elif self.rsi_signal == -1:
            adjusted_mid_price = mid_price * Decimal(1 - (self.bid_spread * natr * self.rsi_spread_change_percentage))
        else:
            adjusted_mid_price = mid_price

        buy_price = adjusted_mid_price * Decimal(1 - self.bid_spread * natr)
        sell_price = adjusted_mid_price * Decimal(1 + self.ask_spread * natr)

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(self.order_amount), price=buy_price)

        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(self.order_amount), price=sell_price)

        return [buy_order, sell_order]

    def get_natr(self):
        candles_df = self.market_data_provider.get_candles_df(self.exchange, self.trading_pair, self.interval,
                                                              self.max_records)
        natr = ta.natr(high=candles_df["high"], low=candles_df["low"], close=candles_df["close"],
                       length=self.natr_period)
        return natr.iloc[-1] / 100

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
