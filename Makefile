.PHONY: install test lint ingest clean-data features stage1 stage2 pipeline evaluate report poster dashboard-data dashboard dashboard-image

install:
	pip install -e ".[dev,dl,app]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

ingest:
	python -m f1pit.ingest

clean-data:
	python -m f1pit.clean

features:
	python -m f1pit.features

stage1:
	python -m f1pit.stage1

stage2:
	python -m f1pit.stage2

pipeline: ingest clean-data features stage1 stage2

# Final test on the held-out seasons. Deliberately NOT part of `pipeline`: run it once, when the
# models are final. A second run on the same models only shows the saved results.
evaluate:
	python -m f1pit.evaluate

# Tables and data facts for the written report, from the saved results -> results/report/
report:
	python -m f1pit.report

# One-page race analysis poster -> results/figures/race_poster_<race>.png   (make poster RACE=2025_05)
poster:
	python -m f1pit.race_poster $(if $(RACE),--race $(RACE),--list)

# Dashboard: export the final results to app/data, then run the app
dashboard-data:
	python -m f1pit.dashboard

dashboard:
	streamlit run app/streamlit_app.py

dashboard-image:
	docker build -f Dockerfile.dashboard -t f1pit-dashboard .
