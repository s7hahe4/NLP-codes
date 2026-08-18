import zipfile
import os

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

print("Creating AD_COG_Kaggle_Data.zip...")
with zipfile.ZipFile('AD_COG_Kaggle_Data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    print("Adding distilbert_baseline...")
    zipdir('distilbert_baseline', zipf)
    print("Adding high_density_dataset...")
    zipdir('high_density_dataset', zipf)
    print("Adding generic_dataset...")
    zipdir('generic_dataset', zipf)
    print("Adding failure_set_top500.json...")
    zipf.write('failure_set_top500.json')

print("Done! AD_COG_Kaggle_Data.zip is ready.")
