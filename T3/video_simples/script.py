# Comando utilizado para extrair os frames do vídeo:
# ffmpeg -i input_video.mp4 ./pics/frame_%04d.png

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Passo C
grayscale_folder = "grayscale_pics"
os.makedirs(grayscale_folder, exist_ok=True)

for frame_file in sorted(os.listdir("pics")):
    frame_path = f"pics/{frame_file}"
    frame = cv2.imread(frame_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{grayscale_folder}/{frame_file}", gray_frame)

# Passo D
template = cv2.imread('template.png', 0)

# Passo E
methods = [
    ('cv2.TM_CCOEFF', cv2.TM_CCOEFF),
    ('cv2.TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
    ('cv2.TM_CCORR', cv2.TM_CCORR),
    ('cv2.TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
    ('cv2.TM_SQDIFF', cv2.TM_SQDIFF),
    ('cv2.TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
]

frames_folder = 'grayscale_pics'

for method_name, method in methods:
    results = [] 

    for frame_file in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path, 0)
        # template matching
        res = cv2.matchTemplate(frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        results.append({'Frame': frame_file, 'min_val': min_val, 'max_val': max_val})
    
    # CSV
    df = pd.DataFrame(results)
    df.to_csv(f'match_results_{method_name}.csv', index=False)

# Passo F
directory_path = 'experimentos'
all_files = os.listdir(directory_path)
csv_files = [file for file in all_files if file.endswith('.csv')]
csv_files.sort()

for index, file_name in enumerate(csv_files, 1):
    plt.figure(figsize=(15, 5))
    df = pd.read_csv(os.path.join(directory_path, file_name))
    frame_numbers = df.index
    method = file_name.replace('match_results_', '').replace('.csv', '')
    plt.plot(frame_numbers, df['min_val'], label=f'Min Value', color='blue')
    plt.plot(frame_numbers, df['max_val'], label=f'Max Value', color='red')
    plt.title(f'{method}')
    plt.xlabel('Frame Number')
    plt.ylabel('Matching Score')
    plt.legend(loc='best')  
    plt.tight_layout() 
    plt.savefig(f'experimentos/grafico_{method}.png')
    plt.show()

# Passo H
frames_dir = 'pics' 
template_path = 'template.png'  
output_video_path = 'output_video.mp4'  

template = cv2.imread(template_path, 0)
template_height, template_width = template.shape[:2]

# Começa a anotar as imagens
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writer = None  

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # template matching
    result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    if video_writer is None:
        frame_height, frame_width = frame.shape[:2]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
    video_writer.write(frame)

if video_writer is not None:
    video_writer.release()

print("Video processing completed.")
