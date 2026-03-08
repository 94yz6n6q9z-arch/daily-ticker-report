name: Test Finnhub

on:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Run Finnhub diagnostic
        run: python test_finnhub.py
        env:
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
