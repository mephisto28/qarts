

class FactorNames:

    # daily 
    PRICE_DEV_FROM_MA = 'price_dev_from_ma'
    PRICE_DEV_FROM_VWAP = 'price_dev_from_vwap'
    PRICE_DEV_FROM_YEST_VWAP = 'price_dev_from_yest_vwap'
    PRICE_POSITION = 'price_position'
    DAILY_MOM = 'daily_mom'
    DAILY_MOM_SUM = 'daily_mom_sum'
    DAILY_VOLATILITY = 'daily_volatility'
    DAILY_VOLVOL = 'daily_volvol'
    DAILY_SKEWNESS = 'daily_skewness'
    DAILY_KURTOSIS = 'daily_kurtosis'

    VOLUME_RATIO = 'volume_ratio'
    RV_CORR = 'rv_corr'
    
    # intraday
    INTRADAY_MOM = 'intraday_mom'
    TODAY_MOM = 'today_mom'
    TODAY_VOLATILITY = 'today_volatility'
    TODAY_SKEWNESS = 'today_skewness'
    TODAY_POSITION = 'today_position'

    # transform
    ABS_TRANSFORM = 'abs'

    # selction
    DAILY_RECENT_VACANCY = 'daily_recent_vacancy'
    IS_UP_LIMIT = 'is_up_limit'
    IS_DOWN_LIMIT = 'is_down_limit'
    IS_TRADABLE = 'is_tradable'

    # targets
    FUTURE_DAY_TARGETS = 'future_day_targets'
    TODAY_TARGETS = 'today_targets'
    FUTURE_DAY_RANGE_TARGETS = 'future_day_range_targets'
    FUTURE_DAY_UP_RANGE_TARGETS = 'future_day_up_range_targets'
    FUTURE_DAY_DOWN_RANGE_TARGETS = 'future_day_down_range_targets'
    RANK_TARGETS = 'rank_targets'
    