# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4313, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4063, in main
    all_signals.extend(compute_signals_for_ticker(t, df))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3254, in compute_signals_for_ticker
    prefix, dist_atr = _classify_vs_level(close, cand.level, atr_val, cand.direction, vol_ratio, clv)
    ^^^^^^^^^^^^^^^^
TypeError: cannot unpack non-iterable NoneType object

```
