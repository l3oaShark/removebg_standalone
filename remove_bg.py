import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Load the model
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
torch.set_float32_matmul_precision("high")
model.to("cuda")
model.eval()

# Transform settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Functions
def select_files():
    global file_paths
    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    lbl_selected.config(text=f"{len(file_paths)} file(s) selected")
    lbl_progress.config(text=f"{"0"}/{len(file_paths)}")

def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if output_folder:
        lbl_output_folder.config(text=f"Output Folder: {output_folder}")

def remove_bg():
    if not file_paths:
        messagebox.showerror("Error", "Please select images first!")
        return
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder!")
        return

    # Create output folder with today's date
    today_folder = os.path.join(output_folder, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(today_folder, exist_ok=True)

    progress_bar["maximum"] = len(file_paths)
    completed = 0

    for file_path in file_paths:
        try:
            image = Image.open(file_path).convert("RGB")
            input_images = transform_image(image).unsqueeze(0).to("cuda")

            with torch.no_grad():
                preds = model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()

            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image.size)
            image.putalpha(mask)

            # แปลงเป็น RGB ถ้าต้องการบันทึกเป็น JPEG
            image_rgb = image.convert("RGB")
            output_path = os.path.join(today_folder, os.path.basename(file_path))

            # บันทึกผลลัพธ์
            if output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg"):
                image_rgb.save(output_path)  # ใช้ RGB สำหรับ JPEG
            else:
                image.save(output_path)  # ใช้ RGBA สำหรับ PNG

            completed += 1
            progress_bar["value"] = completed
            lbl_progress.config(text=f"{completed}/{len(file_paths)} completed")
            root.update_idletasks()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


    messagebox.showinfo("Completed", "Background removal completed!")

# GUI Setup
root = tk.Tk()
root.title("Remove Background Tool")
root.geometry("500x400")

file_paths = []
output_folder = ""

# Widgets
btn_select_files = tk.Button(root, text="Select Images", command=select_files)
btn_select_files.pack(pady=10)

lbl_selected = tk.Label(root, text="No files selected")
lbl_selected.pack()

btn_output_folder = tk.Button(root, text="Select Output Folder", command=select_output_folder)
btn_output_folder.pack(pady=10)

lbl_output_folder = tk.Label(root, text="No output folder selected")
lbl_output_folder.pack()

btn_start = tk.Button(root, text="Start Removing Background", command=remove_bg)
btn_start.pack(pady=20)

progress_bar = ttk.Progressbar(root, length=400, mode="determinate")
progress_bar.pack(pady=10)

lbl_progress = tk.Label(root, text="0/0 completed")
lbl_progress.pack()

# Run
root.mainloop()
