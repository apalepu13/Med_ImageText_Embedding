#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = Med_ImageText_Embedding
PYTHON_INTERPRETER = python3

# Set up the directory structure
PROCESSED_DATA_DIR = data/processed
RAW_DATA_DIR = data/raw
PREPROCESS_SRC_DIR = src/data/preprocessing

# Make tensorflow be quiet
export TF_CPP_MIN_LOG_LEVEL=3

all: preprocess

#################################################################################
# Setup requirements                                                            #
#################################################################################
requirements: python_recs

python_recs:
	$(PYTHON_INTERPRETER) -m pip install --user -r python_requirements.txt

#################################################################################
# Preprocessing                                                                 #
#################################################################################
preprocess: MIMIC_CHX_Images \
            MIMIC_CHX_Reports

MIMIC_CHX_Images: $(PROCESSED_DATA_DIR)/MIMIC_CHX_Images.csv
$(PROCESSED_DATA_DIR)/MIMIC_CHX_Images.csv: $(PREPROCESS_SRC_DIR)/preprocess_MIMIC_CHX_Images.py \
										 $(RAW_DATA_DIR)/MIMIC_CHX_Images.csv
    $(PYTHON_INTERPRETER) $(PREPROCESS_SRC_DIR)/preprocess_MIMIC_CHX_Images.py \
    $(RAW_DATA_DIR)/MIMIC_CHX_Images.csv \
    $(PROCESSED_DATA_DIR)

MIMIC_CHX_Reports: $(PROCESSED_DATA_DIR)/MIMIC_CHX_Reports.csv
$(PROCESSED_DATA_DIR)/MIMIC_CHX_Reports.csv: $(PREPROCESS_SRC_DIR)/preprocess_MIMIC_CHX_Reports.py \
										 $(RAW_DATA_DIR)/MIMIC_CHX_Reports.csv
    $(PYTHON_INTERPRETER) $(PREPROCESS_SRC_DIR)/preprocess_MIMIC_CHX_Reports.py \
    $(RAW_DATA_DIR)/MIMIC_CHX_Reports.csv \
    $(PROCESSED_DATA_DIR)



