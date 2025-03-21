import pandas as pd
import google.generativeai as genai
import time
import os

# Configure the Gemini API
API_KEY = "Your Gemini API Key"  # Replace with your Gemini API key
genai.configure(api_key=API_KEY)

# Define the annotation labels
LABELS = ["Deep Learning", "Computer Vision", "Reinforcement Learning", "NLP", "Optimization"]

# CSV file with scraped data
CSV_FILE = "neurips_2020_2024_data.csv"
OUTPUT_CSV_FILE = "neurips_2020_2024_data_annotated.csv"

def classify_paper(title, abstract):
    """Classify a paper into one of the predefined categories using the Gemini API."""
    try:
        # Create the prompt
        prompt = (
            f"Classify the following research paper into one of these categories: {', '.join(LABELS)}.\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Return only the category name (e.g., 'Deep Learning'). If the paper doesn't fit any category, return 'Other'."
        )

        # Use the Gemini API to generate a response
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use the appropriate model
        response = model.generate_content(prompt)
        label = response.text.strip()

        # Validate the label
        if label not in LABELS:
            label = "Other"
        
        print(f"Classified paper '{title}' as '{label}'")
        return label
    except Exception as e:
        print(f"Error classifying paper '{title}': {e}")
        return "Other"

def annotate_papers():
    """Annotate all papers in the CSV file with a category label."""
    # Load the CSV file
    if not os.path.exists(CSV_FILE):
        print(f"CSV file {CSV_FILE} not found. Please run the scraper first.")
        return
    
    df = pd.read_csv(CSV_FILE)
    
    # Check if the 'label' column already exists; if not, add it
    if "label" not in df.columns:
        df["label"] = "N/A"
    
    # Annotate each paper
    for idx, row in df.iterrows():
        if row["label"] == "N/A" or pd.isna(row["label"]):  # Only annotate if not already labeled
            title = row["title"]
            abstract = row["abstract"]
            label = classify_paper(title, abstract)
            df.at[idx, "label"] = label
            time.sleep(1)  # Be polite to the API with a delay
        
        # Save progress incrementally
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Saved progress to {OUTPUT_CSV_FILE} after processing {idx + 1}/{len(df)} papers")

def main():
    annotate_papers()
    print(f"Annotation complete. Annotated CSV saved to {OUTPUT_CSV_FILE}")

if __name__ == "__main__":
    main()