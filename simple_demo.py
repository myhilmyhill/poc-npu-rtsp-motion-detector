"""
ONNX Runtime QNN Direct Demo
modelsモジュールを使用せず、直接onnxruntime-qnnを使用するデモ
"""
import cv2
import numpy as np
import onnxruntime as ort
import sys
import time
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote

# COCO dataset class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess(frame, input_size=(640, 640)):
    original_height, original_width = frame.shape[:2]
    
    # Resize while maintaining aspect ratio
    scale = min(input_size[0] / original_width, input_size[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((input_size[1], input_size[0], 3), 114, dtype=np.uint8)
    
    # Calculate padding
    pad_x = (input_size[0] - new_width) // 2
    pad_y = (input_size[1] - new_height) // 2
    
    # Place resized image in center
    padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    
    # Convert to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Transpose to (C, H, W)
    transposed = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    batched = np.expand_dims(transposed, axis=0)
    
    return batched, scale, (pad_x, pad_y)

def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def postprocess(output, scale, padding, conf_threshold=0.5, iou_threshold=0.45):
    # YOLOv8 output shape: (1, 84, 8400) -> transpose to (8400, 84)
    predictions = np.squeeze(output[0]).T
    
    # Extract boxes and scores
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    # Get class with max score for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(class_ids)), class_ids]
    
    # Filter by confidence threshold
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # Convert from center format to corner format
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Apply NMS
    indices = nms(boxes_xyxy, confidences, iou_threshold)
    
    boxes_xyxy = boxes_xyxy[indices]
    confidences = confidences[indices]
    class_ids = class_ids[indices]
    
    # Scale back to original size
    pad_x, pad_y = padding
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale
    
    results = []
    for box, conf, class_id in zip(boxes_xyxy, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        results.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": conf,
            "class_name": class_name,
            "class_id": class_id
        })
        
    return results

def main():
    print("=" * 60)
    print("Simple NPU Detection Demo (Direct ORT)")
    print("=" * 60)
    
    print(f"ORT Location: {ort.__file__}")
    print(f"Available Providers: {ort.get_available_providers()}")
    
    model_path = "models/yolov8n.onnx"
    if not Path(model_path).exists():
        print(f"❌ モデルが見つかりません: {model_path}")
        return

    # QNN Options
    qnn_options = {
        'backend_path': 'QnnHtp.dll',
        'qnn_context_priority': 'high',
        'profiling_level': 'basic'
    }
    
    print("\n📦 モデルを読み込み中...")
    try:
        # Explicitly request QNN
        session = ort.InferenceSession(
            model_path,
            providers=[
                ('QNNExecutionProvider', qnn_options),
                'CPUExecutionProvider'
            ]
        )
        print("✅ セッション作成成功")
        print(f"Active Providers: {session.get_providers()}")
        
        if 'QNNExecutionProvider' in session.get_providers():
            print("🚀 QNN (NPU) が有効です！")
        else:
            print("⚠️ QNNが無効です。CPUで実行されます。")
            
    except Exception as e:
        print(f"❌ セッション作成エラー: {e}")
        return

    input_name = session.get_inputs()[0].name
    
    # Input selection
    print("\n入力ソースを選択してください:")
    print("1. Webカメラ (デフォルト)")
    print("2. 画像ファイル")
    print("3. 動画ファイル")
    print("4. RTSPストリーム")
    
    choice = input("\n選択 (1-4, Enter=1): ").strip() or "1"
    
    cap = None
    if choice == "1":
        cap = cv2.VideoCapture(0)
    elif choice == "2":
        img_path = input("画像ファイルのパスを入力: ").strip()
        frame = cv2.imread(img_path)
        if frame is None:
            print("❌ 画像読み込み失敗")
            return
        # Process single image
        start = time.time()
        input_tensor, scale, padding = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        detections = postprocess(outputs, scale, padding)
        end = time.time()
        print(f"推論時間: {(end - start)*1000:.2f} ms")
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    elif choice == "3":
        video_path = input("動画ファイルのパスを入力: ").strip()
        cap = cv2.VideoCapture(video_path)
    elif choice == "4":
        load_dotenv()
        rtsp_url = os.getenv("RTSP_URL")
        username = os.getenv("RTSP_USERNAME")
        password = os.getenv("RTSP_PASSWORD")
        
        if not rtsp_url:
            print("❌ .envにRTSP_URLが設定されていません")
            return
            
        # Construct URL with credentials if provided and not already present
        if username and password and rtsp_url.startswith("rtsp://") and "@" not in rtsp_url:
            # URL encode credentials to handle special characters
            safe_username = quote(username, safe='')
            safe_password = quote(password, safe='')
            
            rtsp_url = rtsp_url.replace("rtsp://", f"rtsp://{safe_username}:{safe_password}@")
            # Mask password for display
            display_url = rtsp_url.replace(f":{safe_password}@", ":****@")
            print(f"📡 RTSP URL from .env: {display_url}")
        else:
            print(f"📡 RTSP URL from .env: {rtsp_url}")
            
        cap = cv2.VideoCapture(rtsp_url)
    
    if cap is None or not cap.isOpened():
        print("❌ 入力を開けませんでした")
        return
        
    print("\n🚀 処理開始 (qキーで終了)...")
    
    frame_count = 0
    total_inference_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if choice == "3":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
            
        # Inference
        start = time.time()
        input_tensor, scale, padding = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        inference_time = time.time() - start
        
        # Postprocess
        detections = postprocess(outputs, scale, padding)
        
        # Draw
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Stats
        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f} (Inf: {inference_time*1000:.1f}ms)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if 'QNNExecutionProvider' in session.get_providers():
             cv2.putText(frame, "Device: NPU (QNN)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        else:
             cv2.putText(frame, "Device: CPU", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("NPU Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
