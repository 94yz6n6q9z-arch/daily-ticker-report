# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4991, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4977, in main
    md.append(build_watchlist_performance_section_md(ohlcv, sector_resolver))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4294, in build_watchlist_performance_section_md
    "Sector": cat,
              ^^^
NameError: name 'cat' is not defined

```
