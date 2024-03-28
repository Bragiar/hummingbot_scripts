import os
from decimal import Decimal
from typing import Dict, List, Set, Optional

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import TradeType, OrderType, PriceType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.smart_components.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.smart_components.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

import pandas_ta as ta


class RsiAmmV2Config(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    candles_config: List[CandlesConfig] = []
    markets: Dict[str, Set[str]] = {}
    connector: str = Field(
        default="binance",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the connector: ",
            prompt_on_new=True
        ))
    trading_pair: str = Field(
        default="BTC-USDT",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the first trading pair: ",
            prompt_on_new=True
        ))
    rsi_period: int = Field(
        default=14, gt=0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI period (e.g. 14): ",
            prompt_on_new=True))
    rsi_low: float = Field(
        default=30, gt=0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI low (e.g. 30): ",
            prompt_on_new=True))
    rsi_high: float = Field(
        default=70, gt=0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI high (e.g. 70): ",
            prompt_on_new=True))
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the interval (e.g. 1m): ",
            prompt_on_new=True))
    order_amount_quote: Decimal = Field(
        default=30, gt=0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the amount of quote asset to be used per order (e.g. 30): ",
            prompt_on_new=True))

    max_records = 1000
    bid_spread = 1  # NATR
    ask_spread = 1  # NATR
    rsi_refresh_time = 15
    create_timestamp = 0
    exchange = "binance"
    TICK = 1

    buffer_size = 100
    rsi_spread_change_percentage = 0.5


class RsiAmmV2(StrategyV2Base):
    account_config_set = False
    rsi_signal = 0

    @classmethod
    def init_markets(cls, config: RsiAmmV2Config):
        """
        Initialize the markets that the strategy is going to use. This method is called when the strategy is created in
        the start command. Can be overridden to implement custom behavior.
        """
        cls.markets = {config.connector: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: RsiAmmV2Config):
        super().__init__(connectors, config)
        self.config = config

        self.market_data_provider = MarketDataProvider(connectors)
        self.max_records = config.rsi_period + 10
        self.last_mid_price = None
        self.market_data_provider.initialize_candles_feed(
            config=CandlesConfig(connector=config.connector,
                                 trading_pair=config.trading_pair,
                                 interval=config.interval,
                                 max_records=config.max_records)
        )



    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        if not self.market_data_provider.ready:
            return []

        new_mid_price = self.market_data_provider.get_price_by_type(self.config.exchange, self.config.exchange, PriceType.MidPrice)

        if self.create_timestamp <= self.current_timestamp:
            # calculate RSI, if signal is 1 then BUY signal, -1 is SELL signal. Shift the MID price accordingly
            self.rsi_signal = self.get_signal()
            self.create_timestamp = self.rsi_refresh_time + self.current_timestamp

        if self.last_mid_price is None or abs(new_mid_price - self.last_mid_price) > self.config.buffer_size:
            # cancel, propose, and place orders
            msg = (f"Mid price is {new_mid_price}")
            self.logger().info(msg)
            #self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal(new_mid_price)
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)

            # self.place_orders(proposal_adjusted)
            self.last_mid_price = new_mid_price

            order1 = proposal_adjusted.pop(0)
            order2 = proposal_adjusted.pop(0)

            executor_config_buy = PositionExecutorConfig(
                timestamp=self.current_timestamp,
                trading_pair=self.config.trading_pair,
                connector_name=self.config.connector,
                side=order1.order_side,
                amount=order1.amount,
            )
            executor_config_sell = PositionExecutorConfig(
                timestamp=self.current_timestamp,
                trading_pair=self.config.trading_pair,
                connector_name=self.config.connector,
                side=order2.order_side,
                amount=order2.amount,
            )
            return [CreateExecutorAction(executor_config=executor_config_buy),
                    CreateExecutorAction(executor_config=executor_config_sell)]
        return []

    def get_signal(self, connector_name: str, trading_pair: str) -> Optional[float]:
        candles = self.market_data_provider.get_candles_df(connector_name, trading_pair, self.config.interval, self.max_records)
        candles.ta.rsi(length=self.config.rsi_period, append=True)
        candles["signal"] = 0
        candles.loc[candles[f"RSI_{self.config.rsi_period}"] < self.config.rsi_low, "signal"] = 1
        candles.loc[candles[f"RSI_{self.config.rsi_period}"] > self.config.rsi_high, "signal"] = -1
        return candles.iloc[-1]["signal"] if not candles.empty else None
    def create_proposal(self, mid_price) -> List[OrderCandidate]:
        natr = self.get_natr()
        if self.rsi_signal == 1:
            adjusted_mid_price = mid_price * Decimal(1 + (self.config.bid_spread * natr * self.config.rsi_spread_change_percentage))
        elif self.rsi_signal == -1:
            adjusted_mid_price = mid_price * Decimal(1 - (self.config.bid_spread * natr * self.config.rsi_spread_change_percentage))
        else:
            adjusted_mid_price = mid_price

        buy_price = adjusted_mid_price * Decimal(1 - self.config.bid_spread * natr)
        sell_price = adjusted_mid_price * Decimal(1 + self.config.ask_spread * natr)

        buy_order = OrderCandidate(trading_pair=self.config.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(self.config.order_amount_quote), price=buy_price)

        sell_order = OrderCandidate(trading_pair=self.config.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(self.config.order_amount_quote), price=sell_price)

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.config.connector].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted
    def get_natr(self):
        candles_df = self.market_data_provider.get_candles_df(self.config.exchange, self.config.trading_pair, self.config.interval,
                                                              self.max_records)
        natr = ta.natr(high=candles_df["high"], low=candles_df["low"], close=candles_df["close"],
                       length=self.config.natr_period)
        return natr.iloc[-1] / 100
    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        return []

    def apply_initial_setting(self):
        for connector in self.connectors.values():
            if self.is_perpetual(connector.name):
                connector.set_position_mode(self.config.position_mode)
                for trading_pair in self.market_data_provider.get_trading_pairs(connector.name):
                    connector.set_leverage(trading_pair, self.config.leverage)
