from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import base64
from PIL import Image  # 导入 PIL 库

app = Flask(__name__)
app = Flask(__name__, static_folder='static')
# 加载模型
model_path = 'best_model.keras'  # 或者 resnet50_model.keras，根据实际情况选择加载哪个模型
model = keras.models.load_model(model_path)

# 初始化 Label Encoder (确保与训练时使用的类别一致)
label_encoder = LabelEncoder()
classes = ['baihe', 'dangshen', 'gouqi', 'huaihua', 'jinyinhua']  # 替换为实际类别
label_encoder.fit(classes)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    if file:
        # 使用 BytesIO 将文件转换为类似文件的对象
        img_stream = BytesIO(file.read())
        
        # 将文件读取为 PIL 图像
        img = image.load_img(img_stream, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # 预测
        prediction = model.predict(img_array)
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        pred_score = np.max(prediction)

        # 将预测图像转换为 base64 编码的字符串
        buffered = BytesIO()
        img.save(buffered, format="JPEG")  # 使用 PIL 的 save 方法
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'class': pred_label, 'confidence': float(pred_score), 'image': img_base64})
    else:
        return jsonify({'error': 'Error processing file.'}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)