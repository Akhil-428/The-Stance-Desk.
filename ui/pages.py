"""Compatibility shim for the former five-page layout.

The redesigned app has two views (ui/predict_view.py, ui/results_view.py);
these names keep older imports working.
"""
from ui import predict_view, results_view

predict_page = predict_view.render
results_page = results_view.render
PAGES: dict = {}
