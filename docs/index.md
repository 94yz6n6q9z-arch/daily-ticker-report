# Daily Report

## ERROR

The run crashed. Traceback:

```text
Traceback (most recent call last):
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 7098, in <module>
    main()
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 6437, in main
    save_state(state)
  File "/home/runner/work/daily-ticker-report/daily-ticker-report/scan.py", line 372, in save_state
    write_text(STATE_PATH, json.dumps(state, indent=2, ensure_ascii=False))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
          ^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 202, in encode
    chunks = list(chunks)
             ^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  [Previous line repeated 1 more time]
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 326, in _iterencode_list
    yield from chunks
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type Timestamp is not JSON serializable

```
