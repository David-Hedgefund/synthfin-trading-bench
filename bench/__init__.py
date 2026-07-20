"""SynthFin Trading Bench — a contamination-free sequential trading benchmark for LLM agents.

The benchmark runs an LLM (or a baseline strategy) as a portfolio manager over a fully
synthetic market scenario. At each rebalance date the agent sees a strictly point-in-time
view of the market (prices, fundamentals, news, filings dated on or before the decision day)
and outputs a target portfolio. Orders fill at the next session's open. Because every
scenario is machine-generated and never published, no model can have seen it in training —
the benchmark is contamination-free by construction.
"""

__version__ = "0.1.0"
