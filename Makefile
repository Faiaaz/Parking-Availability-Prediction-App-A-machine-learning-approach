.PHONY: data features train app all

data:
	python scripts/download_data.py

features:
	python src/data_prep.py

train:
	python src/train.py

app:
	streamlit run app/streamlit_app.py

all: features train app
