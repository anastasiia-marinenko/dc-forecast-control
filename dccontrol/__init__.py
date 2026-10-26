"""Forecast-based control of servers and cooling in a small two-zone data hall.

data.py       - read the Alibaba 2018 trace, make per-minute load of each hall
forecast.py   - persistence, ridge and GRU forecasters, asymmetric loss
hall.py       - decision module + hall simulator
calibrate.py  - pick the server reserve m and the temperature setback s
"""
