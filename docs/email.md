# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4990, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4725, in main
    all_signals.extend(compute_signals_for_ticker(t, df))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3677, in compute_signals_for_ticker
    candidates = detect_pattern_candidates(d)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3414, in detect_pattern_candidates
    dcb = detect_dead_cat_bounce(df)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3375, in detect_dead_cat_bounce
    "gap_strict": bool(strict_gap),
                       ^^^^^^^^^^
NameError: name 'strict_gap' is not defined

```
