# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3340, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 3335, in main
    print(f"Universe={len(universe)}  Signals: early={len(early_sorted)} triggered={len(triggered_sorted)}")
                          ^^^^^^^^
NameError: name 'universe' is not defined

```
