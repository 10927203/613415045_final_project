import cv2
import os

def photos_to_video(image_folder, video_name, fps):
    # 獲取資料夾中所有圖片檔案
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 確保資料夾中有圖片
    if not images:
        print("資料夾中沒有可用的圖片")
        return

    # 獲取圖片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 初始化影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # 將每張圖片加入影片
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 釋放資源
    video.release()
    print(f"影片已儲存為 {video_name}")

# 使用範例
image_folder = 'runs/track/exp4'  # 圖片資料夾路徑
video_name = 'output_video.mp4'             # 輸出的影片名稱
fps = 10                                    # 幀率 (Frames Per Second)

photos_to_video(image_folder, video_name, fps)
