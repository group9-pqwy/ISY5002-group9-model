from flask import Flask, request, jsonify
from PIL import Image
import os
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
import google.generativeai as genai
import cv2
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# 配置 API 密钥
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 创建模型配置
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# 创建生成模型
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a dog breed recognition chatbot that recognizes dog breeds and you can also chat about pet related topics. You have a very friendly tone.",
)

# 初始化聊天会话
chat_session = model.start_chat(history=[])

# 定义 idx2breed 和 all_breeds 字典（确保索引和品种名称对应正确）
idx2breed = {0: 'affenpinscher', 1: 'afghan_hound', 2: 'african_hunting_dog', 3: 'airedale', 4: 'american_staffordshire_terrier', 5: 'appenzeller', 6: 'australian_terrier', 7: 'basenji', 8: 'basset', 9: 'beagle', 10: 'bedlington_terrier', 11: 'bernese_mountain_dog', 12: 'black', 13: 'blenheim_spaniel', 14: 'bloodhound', 15: 'bluetick', 16: 'border_collie', 17: 'border_terrier', 18: 'borzoi', 19: 'boston_bull', 20: 'bouvier_des_flandres', 21: 'boxer', 22: 'brabancon_griffon', 23: 'briard', 24: 'brittany_spaniel', 25: 'bull_mastiff', 26: 'cairn', 27: 'cardigan', 28: 'chesapeake_bay_retriever', 29: 'chihuahua', 30: 'chow', 31: 'clumber', 32: 'cocker_spaniel', 33: 'collie', 34: 'curly', 35: 'dandie_dinmont', 36: 'dhole', 37: 'dingo', 38: 'doberman', 39: 'english_foxhound', 40: 'english_setter', 41: 'english_springer', 42: 'entlebucher', 43: 'eskimo_dog', 44: 'flat', 45: 'french_bulldog', 46: 'german_shepherd', 47: 'german_short', 48: 'giant_schnauzer', 49: 'golden_retriever', 50: 'gordon_setter', 51: 'great_dane', 52: 'great_pyrenees', 53: 'greater_swiss_mountain_dog', 54: 'groenendael', 55: 'ibizan_hound', 56: 'irish_setter', 57: 'irish_terrier', 58: 'irish_water_spaniel', 59: 'irish_wolfhound', 60: 'italian_greyhound', 61: 'japanese_spaniel', 62: 'keeshond', 63: 'kelpie', 64: 'kerry_blue_terrier', 65: 'komondor', 66: 'kuvasz', 67: 'labrador_retriever', 68: 'lakeland_terrier', 69: 'leonberg', 70: 'lhasa', 71: 'malamute', 72: 'malinois', 73: 'maltese_dog', 74: 'mexican_hairless', 75: 'miniature_pinscher', 76: 'miniature_poodle', 77: 'miniature_schnauzer', 78: 'newfoundland', 79: 'norfolk_terrier', 80: 'norwegian_elkhound', 81: 'norwich_terrier', 82: 'old_english_sheepdog', 83: 'otterhound', 84: 'papillon', 85: 'pekinese', 86: 'pembroke', 87: 'pomeranian', 88: 'pug', 89: 'redbone', 90: 'rhodesian_ridgeback', 91: 'rottweiler', 92: 'saint_bernard', 93: 'saluki', 94: 'samoyed', 95: 'schipperke', 96: 'scotch_terrier', 97: 'scottish_deerhound', 98: 'sealyham_terrier', 99: 'shetland_sheepdog', 100: 'shih', 101: 'siberian_husky', 102: 'silky_terrier', 103: 'soft', 104: 'staffordshire_bullterrier', 105: 'standard_poodle', 106: 'standard_schnauzer', 107: 'sussex_spaniel', 108: 'tibetan_mastiff', 109: 'tibetan_terrier', 110: 'toy_poodle', 111: 'toy_terrier', 112: 'vizsla', 113: 'walker_hound', 114: 'weimaraner', 115: 'welsh_springer_spaniel', 116: 'west_highland_white_terrier', 117: 'whippet', 118: 'wire', 119: 'yorkshire_terrier'}

all_breeds = ['Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua', 'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog', 'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'Shih', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset', 'beagle', 'black', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly', 'dhole', 'dingo', 'flat', 'giant_schnauzer', 'golden_retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound', 'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier', 'soft', 'standard_poodle', 'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wire']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
inception = models.inception_v3(pretrained=True)
fc_inputs = inception.fc.in_features
inception.fc = nn.Sequential(
    nn.Linear(fc_inputs, 2048),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(2048, 120),  # 假设有120个犬种
    nn.LogSoftmax(dim=1)
)
inception.load_state_dict(torch.load('inception_finetuning.pth', map_location=device))
inception.eval()  # 设置模型为评估模式
inception = inception.to(device)

# 定义图像预处理
valid_transform = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], p=1)

# 纯种预测函数
def predict_single_image(image):
    """
    预测单张图像的类别.

    Args:
        image: PIL Image对象.

    Returns:
        predicted_class: 预测的类别索引.
        top_prob: 最高概率值.
        second_top_prob: 次高概率值.
    """
    try:
        # 将PIL图像转换为numpy数组
        img = np.array(image)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = valid_transform(image=img)['image']
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = inception(img)
            probabilities = torch.exp(output)
            predicted_class = torch.argmax(probabilities, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            sorted_probs = np.sort(probabilities)
            top_prob = sorted_probs[-1]
            second_top_prob = sorted_probs[-2]
            print(predicted_class)
        return predicted_class.item(), top_prob, second_top_prob

    except Exception as e:
        print(f"发生错误: {e}")
        return None, None, None

# 混种预测函数
def predict(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = inception(image_tensor)
        # 如果模型有辅助输出，需要提取主输出
        if isinstance(outputs, models.inception.InceptionOutputs):
            outputs = outputs.logits
        # 获取预测得分最高的两个类别
        _, predicted = outputs.topk(2, 1, True, True)
        predicted_indices = predicted.cpu().numpy()[0]
        parent_breeds = [idx2breed.get(idx, "Unknown") for idx in predicted_indices]
    return parent_breeds

# 图像预处理函数
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 的均值
            std=[0.229, 0.224, 0.225]    # ImageNet 的标准差
        )
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    return image_tensor.to(device)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(file).convert('RGB')
    except IOError:
        return jsonify({"error": "Invalid image file"}), 400

    # 执行纯种预测
    predicted_class, top_prob, second_top_prob = predict_single_image(image)
    if predicted_class is not None:
        predicted_label = all_breeds[predicted_class]
        # 计算置信度比率
        if second_top_prob == 0:
            confidence_ratio = float('inf')
        else:
            confidence_ratio = top_prob / second_top_prob

        if confidence_ratio > 1.5:  # 阈值可以根据需要调整
            breed_type = "Purebred"
        else:
            breed_type = "Mixed breed"
    else:
        predicted_label = None
        breed_type = "Unknown"

    # 执行混种预测
    parent_breeds = predict(image)

    # 返回所有结果
    return jsonify({
        "message": "Image processed successfully!",
        "breed_type": breed_type,
        "purebred_prediction": predicted_label,
        "mixed_breed_prediction": parent_breeds
    }), 200

# 定义聊天路由
@app.route('/chat', methods=['POST'])
def recognize_breed():
    # 获取请求中的输入
    input_data = request.json.get('input', '')

    if not input_data:
        return jsonify({"error": "Input is required"}), 400

    # 发送输入到模型
    response = chat_session.send_message(input_data)

    # 返回响应文本
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')