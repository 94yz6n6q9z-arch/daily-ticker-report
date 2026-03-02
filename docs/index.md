# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 6765, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 6059, in main
    all_signals.extend(compute_signals_for_ticker(t, df, state=state, debug=debug))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4643, in compute_signals_for_ticker
    candidates = detect_pattern_candidates(d)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4044, in detect_pattern_candidates
    dcb = detect_dead_cat_bounce(df)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3859, in detect_dead_cat_bounce
    atr_prev = float(A[i - 1]) if np.isfinite(A[i - 1]) else float("nan")
                                              ^
NameError: name 'A' is not defined

```
