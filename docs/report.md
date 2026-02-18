# Daily Report

## ERROR

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 1193, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 1021, in main
    session_lf = filter_movers(session_l).sort_values("pct", ascending=True)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/frame.py", line 8347, in sort_values
    k = self._get_label_or_level_values(by[0], axis=axis)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/generic.py", line 1776, in _get_label_or_level_values
    raise KeyError(key)
KeyError: 'pct'

```
