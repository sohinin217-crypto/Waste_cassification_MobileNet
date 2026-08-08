import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

source_dir = r"d:\Dekstop\archive (3)\images\images"
dest_dir = r"d:\Dekstop\archive (3)\dataset"

# Classification mapping
biodegradable_categories = {
    'cardboard_boxes', 'cardboard_packaging',
    'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags',
    'magazines', 'newspaper', 'office_paper', 'paper_cups'
}

non_biodegradable_categories = {
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'steel_food_cans',
    'disposable_plastic_cutlery', 'plastic_cup_lids', 'plastic_detergent_bottles', 
    'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles', 
    'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles',
    'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars',
    'clothing', 'shoes',
    'styrofoam_cups', 'styrofoam_food_containers'
}

def process_single_image(args):
    src_file_path, dest_file_path = args
    try:
        with Image.open(src_file_path) as img:
            # Convert to RGB (in case of RGBA/grayscale)
            rgb_img = img.convert("RGB")
            # Resize to 128x128
            resized_img = rgb_img.resize((128, 128), Image.Resampling.LANCZOS)
            # Save as JPG
            resized_img.save(dest_file_path, "JPEG", quality=95)
        return True
    except Exception as e:
        return False

def main():
    bio_dest = os.path.join(dest_dir, "biodegradable")
    non_bio_dest = os.path.join(dest_dir, "non_biodegradable")
    os.makedirs(bio_dest, exist_ok=True)
    os.makedirs(non_bio_dest, exist_ok=True)

    categories = sorted(os.listdir(source_dir))
    tasks = []

    for category in categories:
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        # Determine the target folder
        if category in biodegradable_categories:
            target_dir = bio_dest
        elif category in non_biodegradable_categories:
            target_dir = non_bio_dest
        else:
            continue
            
        # Process default and real_world subfolders
        for subfolder in ["default", "real_world"]:
            subfolder_path = os.path.join(category_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for file in files:
                src_file_path = os.path.join(subfolder_path, file)
                # Create a unique filename to prevent overlaps
                dest_file_name = f"{category}_{subfolder}_{file}"
                dest_file_name = os.path.splitext(dest_file_name)[0] + ".jpg"
                dest_file_path = os.path.join(target_dir, dest_file_name)
                tasks.append((src_file_path, dest_file_path))

    print(f"Total images found to process: {len(tasks)}")
    
    # Process images in parallel using multiprocessing
    success_count = 0
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Processing images"))
        success_count = sum(1 for r in results if r)

    print(f"\nProcessing finished. Successfully processed {success_count}/{len(tasks)} images.")
    print(f"Biodegradable count: {len(os.listdir(bio_dest))}")
    print(f"Non-Biodegradable count: {len(os.listdir(non_bio_dest))}")

if __name__ == '__main__':
    main()
