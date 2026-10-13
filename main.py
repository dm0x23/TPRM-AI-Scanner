import os
import logging
from src.pipeline import run_full_pipeline

# Suppress verbose urllib3 retry warnings for a cleaner terminal output
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Define Paths
INPUT_PATH = os.path.join("data", "input", "Meridian_Vendor_Input_2.xlsx")
OUTPUT_PATH = os.path.join("data", "output", "Completed_Meridian_Assessment.xlsx")
CONTACT_EMAIL = "optiv.team@email.com"

if __name__ == "__main__":
    print("Initializing V-OSINT: Meridian AI Vendor Risk Assessment...")

    # Ensure necessary directories exist
    os.makedirs(os.path.join("data", "input"), exist_ok=True)
    os.makedirs(os.path.join("data", "output"), exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        print(
            "Please place the 'Meridian_Vendor_Input_2.xlsx' file in the 'data/input/' directory."
        )
    else:
        run_full_pipeline(INPUT_PATH, OUTPUT_PATH, CONTACT_EMAIL)
