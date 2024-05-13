from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions 
# import base64

# # 画像データのBase64エンコード
# def get_image_base64(img_file):
#     img_file.seek(0)
#     img_data = base64.b64encode(img_file.read()).decode('utf-8')
#     return f'data:image/jpeg;base64,{img_data}'

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            # img_file = form.cleaned_data['image']
            # img_file.seek(0)
            # img_data = get_image_base64(img_file) 
            # img_file.seek(0)  # エンコード後、再度同じ画像ファイルを使用するためシーク位置をリセット
            # img = load_img(img_file, target_size=(224, 224))
            # img_array = img_to_array(img)
            # img_array = img_array.reshape((1, 224, 224, 3))
            # # VGG16モデルの入力形式に合わせて前処理
            img_array = preprocess_input(img_array)
            # モデルを読み込み、予測を行い、結果をテンプレートに渡す
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            preds = model.predict(img_array)
            top_preds = decode_predictions(preds, top=5)[0]
            # 画像と予測結果をテンプレートに渡す
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': top_preds, 'img_data': img_data})
        else:
            # フォームが無効な場合は、再度フォームをユーザーに提示
            return render(request, 'home.html', {'form': form})

