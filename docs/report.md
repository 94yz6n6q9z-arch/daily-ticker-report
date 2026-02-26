# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4991, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4800, in main
    df_early = signals_to_df(early_sorted, sector_resolver=sector_resolver, name_resolver=company_name_for_ticker, country_resolver=country_for_ticker)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 4350, in signals_to_df
    "Sector": sec,
              ^^^
NameError: name 'sec' is not defined

```
