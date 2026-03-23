import os
import pandas as pd

# Paths
BASE_DIR = "Dataset"
SETS = ["train_valid", "test"]

data_entries = []

# Iterate through train_valid and test folders
for dataset_type in SETS:
    dataset_path = os.path.join(BASE_DIR, dataset_type)
    
    for body_part in os.listdir(dataset_path):
        body_part_path = os.path.join(dataset_path, body_part)
        if not os.path.isdir(body_part_path):
            continue
        
        for patient in os.listdir(body_part_path):
            patient_path = os.path.join(body_part_path, patient)
            if not os.path.isdir(patient_path):
                continue
            
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue
                
                label = 1 if "positive" in study.lower() else 0  # 1 = fracture, 0 = no fracture
                
                for img_name in os.listdir(study_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(study_path, img_name)
                        data_entries.append({
                            "dataset": dataset_type,
                            "body_part": body_part,
                            "patient_id": patient,
                            "study": study,
                            "label": label,
                            "image_path": img_path
                        })

# Convert to DataFrame
df = pd.DataFrame(data_entries)
print(f"✅ Total Images Found: {len(df)}")

# Save separate CSVs
for dataset_type in SETS:
    df_subset = df[df["dataset"] == dataset_type]
    output_path = f"{dataset_type}_dataset.csv"
    df_subset.to_csv(output_path, index=False)
    print(f"💾 Saved: {output_path} ({len(df_subset)} samples)")
