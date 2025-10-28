env:
	conda env create -f env.yml

prep:
	python scripts/prep_adata.py --input data/raw --output data/adata/baseline.h5ad --region BA4

slides:
	python scripts/export_figs.py

all: prep

