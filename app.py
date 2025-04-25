import logging
from flask import Flask, request, render_template, send_from_directory
from pathlib import Path
import cv2
import numpy as np
import joblib
import torch
from model.unet_model import UNet
from svm_extraction.feature_extraction import extract_features

# 配置日志记录
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
device = torch.device('cpu')

# 加载 U-Net 模型
try:
    unet_model = UNet(n_channels=1, n_classes=1)
    unet_model.to(device=device)
    unet_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    unet_model.eval()
except Exception as e:
    logging.error(f"加载分割模型失败：{str(e)}")
    unet_model = None
    # 可以考虑在这里直接退出程序
    # import sys
    # sys.exit(1)

# 加载 SVM 模型
try:
    svm_model = joblib.load('trained_svm_model.pkl')
except Exception as e:
    logging.error(f"加载 SVM 模型失败：{str(e)}")
    svm_model = None
    # 可以考虑在这里直接退出程序
    # import sys
    # sys.exit(1)

# 加载标准化器
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    logging.error("未找到标准化器文件 scaler.pkl，请先训练模型并保存标准化器。")
    scaler = None
    # 可以考虑在这里直接退出程序
    # import sys
    # sys.exit(1)

# 创建临时目录
tmp_dir = Path('tmp')
if not tmp_dir.exists():
    tmp_dir.mkdir()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('image')
    if not file:
        return {'status': 'error','message': '未上传文件'}
    try:
        img_path = tmp_dir / file.filename
        file.save(str(img_path))
        return {'status':'success', 'img_path': str(img_path)}
    except Exception as e:
        logging.error(f'上传失败：{str(e)}')
        return {'status': 'error','message': f'上传失败：{str(e)}'}


@app.route('/segment', methods=['POST'])
def segment_disc():
    if not unet_model:
        return {'status': 'error','message': '分割模型未加载'}
    data = request.get_json()
    img_path = data.get('img_path')
    if not img_path:
        return {'status': 'error','message': '未提供图像路径'}
    try:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (512, 512))
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = unet_model(tensor)

        mask = (pred.cpu().numpy()[0][0] > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_path = tmp_dir /'mask.jpg'
        cv2.imwrite(str(mask_path), mask)
        return {'status':'success','mask_path': str(mask_path)}
    except Exception as e:
        logging.error(f'分割失败：{str(e)}')
        return {'status': 'error','message': f'分割失败：{str(e)}'}


@app.route('/predict', methods=['POST'])
def predict_degeneration():
    if not svm_model or not scaler:
        return {'status': 'error','message': '请先训练 SVM 模型并加载标准化器'}
    data = request.get_json()
    img_path = data.get('img_path')
    mask_path = data.get('mask_path')
    if not img_path or not mask_path:
        return {'status': 'error','message': '未提供图像或掩码路径'}
    img_path = Path(img_path)
    mask_path = Path(mask_path)
    if not img_path.exists() or not mask_path.exists():
        return {'status': 'error','message': '未找到图像文件'}
    try:
        mask = cv2.imread(str(mask_path), 0)
        original = cv2.imread(str(img_path))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]
        total = len(valid_contours)

        annotated = original.copy()
        prediction_counts = {'degenerative': 0, 'normal': 0}

        for cnt in valid_contours:
            single_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(single_mask, [cnt], -1, 255, -1)

            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            features = extract_features(single_mask, original_gray)
            features = scaler.transform([features])[0]

            prediction = svm_model.predict([features])
            if prediction[0] == 1:
                prediction_counts['degenerative'] += 1
                color = (0, 0, 255)
            else:
                prediction_counts['normal'] += 1
                color = (0, 255, 0)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, "Degenerative" if prediction[0] == 1 else "Normal",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        result_path = tmp_dir / 'annotated.jpg'
        cv2.imwrite(str(result_path), annotated)
        message = f"分析完成，发现{total}个椎间盘（退变：{prediction_counts['degenerative']}，正常：{prediction_counts['normal']}）"
        return {'status':'success', 'result_path': str(result_path),'message': message,
                'degenerative_count': prediction_counts['degenerative'],
                'normal_count': prediction_counts['normal']}
    except Exception as e:
        logging.error(f'预测失败：{str(e)}')
        return {'status': 'error','message': f'预测失败：{str(e)}'}


@app.route('/tmp/<path:filename>')
def serve_tmp_file(filename):
    return send_from_directory(str(tmp_dir), filename)


if __name__ == '__main__':
    app.run(debug=True)