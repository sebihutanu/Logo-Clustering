import os
import shutil
import pandas as pd

def move_in_clusters(logo_dir, csv_file):
    if csv_file == 'logo_min_cls.csv':
        classification = 'is_minimalist'
    else :
        classification = 'classification'
    df = pd.read_csv(csv_file)
    
    for cluster in df[classification].unique():
        cluster_dir = os.path.join(logo_dir, f"form_{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        cluster_filenames = df[df[classification] == cluster]["filename"].values
        for fname in cluster_filenames:
            src_path = os.path.join(logo_dir, fname)
            dest_path = os.path.join(cluster_dir, fname)
            try:
                shutil.copy(src_path, dest_path)
            except Exception as e:
                print(f"Error copying {fname}: {e}")

if __name__ == "__main__":
    logo_dir = "logos"
    csv_file = "logo_min_cls.csv"
    move_in_clusters(logo_dir=logo_dir, csv_file=csv_file)