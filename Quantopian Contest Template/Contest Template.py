# Import Algorithm API functions
from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio

# Import Optimize API module
import quantopian.optimize as opt

# Pipeline imports
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.factors import SimpleMovingAverage

# Import built-in universe and Risk API method
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline


def initialize(context):
    # Constraint parameters
    context.max_leverage = 1.0
    context.max_short_pos_size = 0.03
    context.max_long_pos_size = 0.03
    context.max_turnover = 0.95

    # Attach data pipelines
    attach_pipeline(make_pipeline(), 'data_pipe')
    attach_pipeline(risk_loading_pipeline(), 'risk_pipe')

    # Schedule rebalance function
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

# Get pipeline outputs and store them in context
def before_trading_start(context, data):
    context.output = pipeline_output('data_pipe')
    context.risk_factor_betas = pipeline_output('risk_pipe')

# Pipeline definition
def make_pipeline():

    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
        mask=QTradableStocksUS()
    )

    return Pipeline(
        columns={
            'sentiment_score': sentiment_score,
        },
        screen=sentiment_score.notnull(),
        screen=QTradableStocksUS()
    )


def rebalance(context, data):
    # Create MaximizeAlpha objective using
    objective = opt.MaximizeAlpha(context.output.sentiment_score)

    # Create position size constraint
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -context.max_short_pos_size,
        context.max_long_pos_size
    )

    # Ensure long and short books are reoughly equal
    dollar_neutral = opt.DollarNeutral()

    # Constrain target portfolio's leverage
    max_leverage = opt.MaxGrossExposure(context.max_leverage)

    # Constrain portfolio turnover
    max_turnover = opt.MaxTurnover(context.max_turnover)

    # Constrain target portfolio's risk exposure | sector = 0.2 style = 0.4
    factor_risk_constraints = opt.experimental.RiskModelExposure(context.risk_factor_betas, version=opt.Newest)

    # Rebalance portfolio using objective and constraints
    order_optimal_portfolio(
        objective=objective,
        constraints=[
            max_leverage,
            dollar_neutral,
            constrain_pos_size,
            max_turnover,
            factor_risk_constraints,
        ]
    )