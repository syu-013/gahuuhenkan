import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import io
import random
from colorsys import rgb_to_hsv, hsv_to_rgb

# --- 1. 画風変換関数（既存 + 強化） ---

def convert_to_pixel_art(image_pil, pixel_size=8, num_colors=64):
    original_width, original_height = image_pil.size
    quantized_image = image_pil.quantize(colors=num_colors)
    small_width = original_width // pixel_size
    small_height = original_height // pixel_size
    temp_image = quantized_image.resize((small_width, small_height), Image.NEAREST)
    pixel_art_image = temp_image.resize((original_width, original_height), Image.NEAREST)
    return pixel_art_image

def convert_to_anime_style(image_pil, line_threshold1=50, line_threshold2=150, line_thickness=1, color_clusters=16, blur_strength=2, shadow_intensity=0.3, highlight_intensity=0.2, pencil_texture_intensity=0.0):
    img_cv = np.array(image_pil.convert('RGB'))
    
    # 1. 色の量子化 (K-means)
    data = img_cv.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = max(2, min(256, color_clusters))  
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    quantized_img_cv = center[label.flatten()].reshape(img_cv.shape)

    # 色面のぼかし
    if blur_strength > 0:
        kernel_size_blur = blur_strength * 2 + 1
        quantized_img_cv = cv2.GaussianBlur(quantized_img_cv, (kernel_size_blur, kernel_size_blur), 0)

    # 2. 線画の抽出と強調 (Canny & Dilate)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred_gray_for_edges = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_gray_for_edges, line_threshold1, line_threshold2)
    if line_thickness > 0:
        kernel_line = np.ones((1,1), np.uint8) 
        edges = cv2.dilate(edges, kernel_line, iterations=line_thickness)

    # 3. 影とハイライトの適用
    hsv_original = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    brightness = hsv_original[:,:,2] / 255.0
    final_color_img = quantized_img_cv.copy()
    
    shadow_mask = (brightness < 0.5).astype(np.float32)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (5,5), 0) 
    for c in range(3):
        final_color_img[:,:,c] = (final_color_img[:,:,c] * (1.0 - shadow_mask * shadow_intensity)).astype(np.uint8)
        
    highlight_mask = (brightness > 0.8).astype(np.float32)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (5,5), 0) 
    for c in range(3):
        final_color_img[:,:,c] = (final_color_img[:,:,c] + (255 - final_color_img[:,:,c]) * highlight_mask * highlight_intensity).astype(np.uint8)
        final_color_img[:,:,c] = np.clip(final_color_img[:,:,c], 0, 255)

    final_output_cv = final_color_img.copy()

    # 5. 線画の重ね合わせ
    final_output_cv[edges == 255] = [0, 0, 0] 

    # 6. 色鉛筆調テクスチャの追加
    if pencil_texture_intensity > 0.0:
        texture = np.random.randint(0, 256, final_output_cv.shape, dtype=np.uint8)
        texture_gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
        
        hsv = cv2.cvtColor(final_output_cv, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:,:,2].astype(np.float32) / 255.0 
        
        texture_effect = (texture_gray.astype(np.float32) / 255.0 - 0.5) * 2.0 
        texture_effect_weighted = texture_effect * v_channel * pencil_texture_intensity * 100 
        
        final_output_float = final_output_cv.astype(np.float32)
        final_output_float += texture_effect_weighted[:,:,np.newaxis] 
        final_output_cv = np.clip(final_output_float, 0, 255).astype(np.uint8)

    return Image.fromarray(final_output_cv)

def convert_to_oil_painting(image_pil, brush_size=5, intensity=20, use_stylization=True):
    img_cv = np.array(image_pil.convert('RGB'))
    
    if use_stylization:
        try:
            sigma_s = brush_size * 5  
            sigma_r = intensity / 255.0  
            stylized_img_cv = cv2.stylization(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), sigma_s=sigma_s, sigma_r=sigma_r)
            return Image.fromarray(cv2.cvtColor(stylized_img_cv, cv2.COLOR_BGR2RGB))
        except AttributeError:
            st.warning("OpenCVの高度な油絵フィルタ（cv2.stylization）は利用できません。代替フィルタを使用します。")
            pass 
            
    height, width, _ = img_cv.shape
    output_img = np.zeros_like(img_cv)
    img_np_quantized = (img_cv // intensity) * intensity if intensity > 0 else img_cv.copy()

    for y in range(0, height, brush_size):
        for x in range(0, width, brush_size):
            patch = img_np_quantized[y:y+brush_size, x:x+brush_size]
            if patch.size == 0:  
                continue
            colors, counts = np.unique(patch.reshape(-1, 3), axis=0, return_counts=True)
            if colors.size == 0:  
                continue
            most_frequent_color = colors[np.argmax(counts)]
            output_img[y:y+brush_size, x:x+brush_size] = most_frequent_color
            
    output_img = cv2.GaussianBlur(output_img, (3, 3), 0)
    return Image.fromarray(output_img)

def convert_to_sketch(image_pil):
    img_cv = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)  
    sketch_img_cv = cv2.divide(gray, 255 - blurred, scale=256.0)
    return Image.fromarray(sketch_img_cv).convert('L')


# --- 2. エフェクト・ぼかし関数（既存） ---

def apply_blur(image_pil, blur_radius):
    if blur_radius <= 0:
        return image_pil
    return image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def apply_sharpen(image_pil, factor):
    if factor <= 1.0:
        return image_pil.filter(ImageFilter.UnsharpMask(radius=0, percent=0, threshold=0))
    return image_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=int(factor * 100), threshold=3))

def apply_sepia(image_pil):
    img_cv = np.array(image_pil.convert('RGB'))
    img_float = img_cv.astype(np.float32) / 255.0
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img_float = np.dot(img_float, sepia_matrix.T)
    sepia_img_cv = np.clip(sepia_img_float * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_img_cv)

def apply_grayscale(image_pil):
    return image_pil.convert('L')  

def apply_invert(image_pil):
    return ImageOps.invert(image_pil.convert('RGB'))  

# --- 3. 特定の色域操作関数（既存） ---

def apply_color_manipulation(image_pil, target_hue, target_saturation, target_value, color_tolerance, sat_val_tolerance, effect_type, new_hue=None):
    img_cv = np.array(image_pil.convert('RGB'))
    hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)

    mask = None
    if target_hue == 0:  
        lower_h1 = 0
        upper_h1 = color_tolerance  
        lower_h2 = 179 - color_tolerance
        upper_h2 = 179
        
        lower_s = max(0, target_saturation - sat_val_tolerance)
        upper_s = min(255, target_saturation + sat_val_tolerance)
        lower_v = max(0, target_value - sat_val_tolerance)
        upper_v = min(255, target_value + sat_val_tolerance)

        mask1 = cv2.inRange(hsv_img, np.array([lower_h1, lower_s, lower_v]), np.array([upper_h1, upper_s, upper_v]))
        mask2 = cv2.inRange(hsv_img, np.array([lower_h2, lower_s, lower_v]), np.array([upper_h2, upper_s, upper_v]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower_h = max(0, target_hue - color_tolerance)
        upper_h = min(179, target_hue + color_tolerance)
        
        lower_s = max(0, target_saturation - sat_val_tolerance)
        upper_s = min(255, target_saturation + sat_val_tolerance)
        lower_v = max(0, target_value - sat_val_tolerance)
        upper_v = min(255, target_value + sat_val_tolerance)
        
        mask = cv2.inRange(hsv_img, np.array([lower_h, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))
    
    result_img = img_cv.copy()

    if effect_type == "彩度強調":
        hsv_img_copy = hsv_img.copy()
        s_channel = hsv_img_copy[:,:,1].astype(np.float32)
        s_channel[mask > 0] = np.clip(s_channel[mask > 0] * 1.5, 0, 255)  
        hsv_img_copy[:,:,1] = s_channel.astype(np.uint8)
        result_img = cv2.cvtColor(hsv_img_copy, cv2.COLOR_HSV2RGB)

    elif effect_type == "色相変換":
        if new_hue is not None:
            hsv_img_copy = hsv_img.copy()
            hsv_img_copy[mask > 0, 0] = new_hue  
            result_img = cv2.cvtColor(hsv_img_copy, cv2.COLOR_HSV2RGB)
        else:
            result_img = image_pil.copy()

    elif effect_type == "モノクロ化（背景）":
        inverse_mask = cv2.bitwise_not(mask)
        gray_background = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        gray_background_rgb = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2RGB)
        
        result_img = cv2.bitwise_and(img_cv, img_cv, mask=mask)
        result_img = cv2.bitwise_or(result_img, cv2.bitwise_and(gray_background_rgb, gray_background_rgb, mask=inverse_mask))
    
    # --- カラーポップ機能は削除されるため、このブロックは不要 ---
    # elif effect_type == "カラーポップ": 
    #     inverse_mask = cv2.bitwise_not(mask)
    #     gray_background = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    #     gray_background_rgb = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2RGB)
        
    #     result_img = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    #     result_img = cv2.bitwise_or(result_img, cv2.bitwise_and(gray_background_rgb, gray_background_rgb, mask=inverse_mask))
    # --- 削除ここまで ---
        
    return Image.fromarray(result_img)

# --- 4. 画像分析・主要色抽出関数（AI: K-means） ---
def extract_dominant_colors(image_pil, num_colors=8):
    """
    画像から主要な色を抽出し、RGB値のリストとして返します。
    """
    img_cv = np.array(image_pil.convert('RGB'))
    small_img = cv2.resize(img_cv, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = num_colors
    
    ret, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_colors = np.uint8(centers)
    return dominant_colors

def analyze_image_features(image_pil):
    """
    画像の色、明るさ、コントラストなどを分析し、特徴量を返します。
    """
    img_cv = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # 明るさ (平均輝度)
    mean_brightness = np.mean(gray)

    # コントラスト (標準偏差)
    contrast = np.std(gray)

    # 色の鮮やかさ (彩度の平均)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    mean_saturation = np.mean(hsv[:,:,1])

    # その他の特徴量（例: エッジの量）
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges / 255.0) / (img_cv.shape[0] * img_cv.shape[1])

    return {
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "mean_saturation": mean_saturation,
        "edge_density": edge_density
    }

def suggest_filters_based_on_features(features):
    """
    分析された特徴量に基づいてフィルターのリストを提案します。
    このロジックはシンプルなルールベースです。
    """
    suggestions = []

    # 明るさに基づいた提案
    if features["mean_brightness"] < 80:
        suggestions.append("シャープ") 
    elif features["mean_brightness"] > 180:
        suggestions.append("ぼかし") 

    # コントラストに基づいた提案
    if features["contrast"] < 40:
        suggestions.append("アニメ風") 
        suggestions.append("彩度強調")
    elif features["contrast"] > 80:
        suggestions.append("スケッチ風") 

    # 彩度に基づいた提案
    if features["mean_saturation"] < 50:
        suggestions.append("セピア") 
        suggestions.append("モノクロ")
    elif features["mean_saturation"] > 150:
        suggestions.append("ドット絵風") 
        suggestions.append("彩度強調")

    # エッジ密度に基づいた提案
    if features["edge_density"] > 0.05:
        suggestions.append("油絵風") 

    return list(set(suggestions))


# --- 6. 新機能: ノスタルジック・フィルター関数 ---

def apply_rgb_channel_shift(image_pil, shift_amount=5):
    img_cv = np.array(image_pil.convert('RGB'))
    height, width, _ = img_cv.shape
    
    shifted_img = np.zeros_like(img_cv)
    
    shifted_img[:, :, 0] = np.roll(img_cv[:, :, 0], shift_amount, axis=1)
    shifted_img[:, :, 1] = img_cv[:, :, 1]
    shifted_img[:, :, 2] = np.roll(img_cv[:, :, 2], -shift_amount, axis=1)
    
    return Image.fromarray(shifted_img)

def apply_glitch_effect(image_pil, intensity=0.1, num_glitches=5):
    img_cv = np.array(image_pil.convert('RGB'))
    height, width, _ = img_cv.shape
    
    glitched_img = img_cv.copy()
    
    for _ in range(num_glitches):
        y_start = random.randint(0, height - 1)
        height_segment = random.randint(1, int(height * intensity))
        y_end = min(height, y_start + height_segment)
        
        shift_x = random.randint(-int(width * intensity), int(width * intensity))
        
        if y_start < y_end:
            segment = glitched_img[y_start:y_end, :, :]
            glitched_img[y_start:y_end, :, :] = np.roll(segment, shift_x, axis=1)
            
    return Image.fromarray(glitched_img)

# --- 8. 新機能: 画像モザイク加工 ---
def apply_mosaic_effect(image_pil, block_size=10):
    img_cv = np.array(image_pil.convert('RGB'))
    height, width, _ = img_cv.shape

    if block_size <= 1:
        return image_pil

    small_img = cv2.resize(img_cv,  
                           (width // block_size, height // block_size),  
                           interpolation=cv2.INTER_LINEAR)
    
    mosaic_img = cv2.resize(small_img,  
                            (width, height),  
                            interpolation=cv2.INTER_NEAREST) 
    
    return Image.fromarray(mosaic_img)

# --- 新機能: クールなスタイル関数 ---

def apply_duotone(image_pil, color1=(0, 0, 0), color2=(255, 255, 255), balance=0.5):
    """画像をデュオトーン（2色調）に変換します。バランスで暗い色と明るい色の比率を調整できます。"""
    img_gray = image_pil.convert('L') # グレースケールに変換
    img_np = np.array(img_gray)       # NumPy配列に変換 (0-255)

    # 輝度値を0-1の範囲に正規化
    normalized_brightness = img_np / 255.0

    # バランス値を考慮したグラデーションマッピング
    # balanceは色の移行の「中心点」を調整します
    
    interpolation_factor = np.zeros_like(normalized_brightness)

    if balance <= 0: # balanceが0以下の場合はすべてcolor1
        interpolation_factor = np.zeros_like(normalized_brightness)
    elif balance >= 1: # balanceが1以上の場合はすべてcolor2
        interpolation_factor = np.ones_like(normalized_brightness)
    else:
        # 輝度値を`power`を使って歪ませることで、balanceの影響を表現
        # balanceが0.5より小さいほどgammaが大きくなり、暗い部分が強調
        # balanceが0.5より大きいほどgammaが小さくなり、明るい部分が強調
        # balanceを0.0-1.0から、gamma値 (例えば0.2から5.0の範囲) にマッピング
        # balanceが0.5のときgamma=1
        # より良いgammaマッピング: balance=0.5でgamma=1、0で大きくなり、1で小さくなる
        # 例: gamma = (1.0 - balance) * 3 + 0.2  # balance=0 -> 3.2, balance=0.5 -> 1.7, balance=1 -> 0.2
        # あるいは、より中心に寄せるために
        gamma_value = 1 / (balance * 1.8 + 0.1) # balanceが0.5なら約1.05、0.1なら約5.2、0.9なら約0.57
        interpolation_factor = np.power(normalized_brightness, gamma_value)

    # interpolation_factor が0-1の範囲に収まるようにクリップ
    interpolation_factor = np.clip(interpolation_factor, 0, 1)

    # 色の線形補間
    r = color1[0] * (1 - interpolation_factor) + color2[0] * interpolation_factor
    g = color1[1] * (1 - interpolation_factor) + color2[1] * interpolation_factor
    b = color1[2] * (1 - interpolation_factor) + color2[2] * interpolation_factor

    duotone_img_np = np.stack([r, g, b], axis=-1)
    duotone_img_np = np.clip(duotone_img_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(duotone_img_np)


def apply_halftone(image_pil, dot_size=5, shape='circle', angle=0, contrast_factor=1.0):
    """画像をハーフトーン（網点）効果に変換します。ドットの形状、角度、コントラストを調整できます。"""
    img_gray = image_pil.convert('L')
    img_np = np.array(img_gray)

    # コントラスト調整
    if contrast_factor != 1.0:
        img_np = np.clip(128 + contrast_factor * (img_np - 128), 0, 255).astype(np.uint8)
        img_gray = Image.fromarray(img_np)

    width, height = img_gray.size
    halftone_img = Image.new('L', (width, height), 255) # 白い背景

    draw = ImageDraw.Draw(halftone_img)

    # 角度をラジアンに変換
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    for y in range(0, height, dot_size):
        for x in range(0, width, dot_size):
            # サンプリング領域の中心
            cx = x + dot_size / 2
            cy = y + dot_size / 2

            box = (x, y, x + dot_size, y + dot_size)
            cropped = img_gray.crop(box)
            
            avg_brightness = np.mean(np.array(cropped))

            # 輝度に基づいてドットのサイズを決定 (明るいほど小さいドット)
            size_factor = (1 - (avg_brightness / 255.0)) # 0 (明るい) -> 1 (暗い)
            current_dot_size = size_factor * dot_size

            if current_dot_size > 0:
                if shape == 'circle':
                    radius = current_dot_size / 2
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=0)
                elif shape == 'square':
                    half_side = current_dot_size / 2
                    points = [
                        (cx - half_side, cy - half_side),
                        (cx + half_side, cy - half_side),
                        (cx + half_side, cy + half_side),
                        (cx - half_side, cy + half_side)
                    ]
                    # 各点を中心(cx, cy)に対してangle度回転
                    rotated_points = []
                    for px, py in points:
                        dx, dy = px - cx, py - cy
                        rotated_x = dx * cos_angle - dy * sin_angle + cx
                        rotated_y = dx * sin_angle + dy * cos_angle + cy
                        rotated_points.append((rotated_x, rotated_y))
                    draw.polygon(rotated_points, fill=0)
                elif shape == 'diamond':
                    points = [
                        (cx, cy - current_dot_size / 2),
                        (cx + current_dot_size / 2, cy),
                        (cx, cy + current_dot_size / 2),
                        (cx - current_dot_size / 2, cy)
                    ]
                    # 各点を中心(cx, cy)に対してangle度回転
                    rotated_points = []
                    for px, py in points:
                        dx, dy = px - cx, py - cy
                        rotated_x = dx * cos_angle - dy * sin_angle + cx
                        rotated_y = dx * sin_angle + dy * cos_angle + cy
                        rotated_points.append((rotated_x, rotated_y))
                    draw.polygon(rotated_points, fill=0)

    return halftone_img

# --- Streamlitアプリの本体 ---
st.set_page_config(layout="wide", page_title="画風変換アプリ")

st.title("画風変換アプリ")
st.write("画像をアップロードして、様々なスタイルやエフェクトを試してみましょう！")
st.markdown("---")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# セッションステートの初期化
if 'target_h' not in st.session_state: st.session_state.target_h = 0
if 'target_s' not in st.session_state: st.session_state.target_s = 0
if 'target_v' not in st.session_state: st.session_state.target_v = 0
if 'initial_color_hex' not in st.session_state: st.session_state.initial_color_hex = "#FF0000"
if 'last_color_tolerance' not in st.session_state: st.session_state.last_color_tolerance = 10
if 'last_sat_val_tolerance' not in st.session_state: st.session_state.last_sat_val_tolerance = 30


# RGB HEXからHSVを計算し、セッションステートを更新するコールバック関数
def update_hsv_from_hex(rgb_hex):
    r, g, b = int(rgb_hex[1:3], 16), int(rgb_hex[3:5], 16), int(rgb_hex[5:7], 16)
    rgb_color_cv = np.uint8([[[b, g, r]]])  
    hsv_color_cv = cv2.cvtColor(rgb_color_cv, cv2.COLOR_BGR2HSV)
    
    st.session_state.target_h = int(hsv_color_cv[0][0][0])
    st.session_state.target_s = int(hsv_color_cv[0][0][1])
    st.session_state.target_v = int(hsv_color_cv[0][0][2])
    st.session_state.initial_color_hex = rgb_hex

# --- メインコンテンツ ---
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.subheader("元の画像")
        st.image(original_image, use_column_width=True)

    processed_image = None
    current_style_or_effect_name = ""

    st.sidebar.subheader("変換タイプを選択")
    transform_type_choice = st.sidebar.selectbox(
        "実行したい操作を選んでください",
        (
            "選択してください",  
            "AIフィルター提案",
            "画風変換",  
            "エフェクト・ぼかし",  
            "特定の色を変換/強調",  
            "レトロ・ノスタルジック風", 
            "画像モザイク加工",
            "スタイル" 
        )
    )
    st.sidebar.markdown("---")

    if transform_type_choice == "AIフィルター提案":
        st.sidebar.subheader("AIのおすすめ")
        st.sidebar.info("画像の特徴をAIが分析し、最適なフィルターを提案します。")
        if st.sidebar.button("おすすめフィルターを分析！"):
            with st.spinner("画像分析中..."):
                features = analyze_image_features(original_image)
                suggested_filters = suggest_filters_based_on_features(features)
                if suggested_filters:
                    st.sidebar.write("**あなたにおすすめのフィルター:**")
                    for filter_name in suggested_filters:
                        st.sidebar.button(filter_name, key=f"suggested_btn_{filter_name}") 
                else:
                    st.sidebar.info("この画像には特に提案できるフィルターがありませんでした。")

    elif transform_type_choice == "画風変換":
        st.sidebar.subheader("画風スタイルを選択")
        style_choice = st.sidebar.selectbox(
            "どの画風スタイルに変換しますか？",
            ("選択してください", "ドット絵風", "アニメ風", "スケッチ風", "油絵風")
        )
        current_style_or_effect_name = style_choice

        if style_choice == "ドット絵風":
            pixel_size = st.sidebar.slider("ドットのサイズ", 2, 50, 8)  
            num_colors_pixel = st.sidebar.slider("ドット絵の色数", 8, 256, 64)
            if st.sidebar.button("ドット絵風に変換"):
                with st.spinner(f"画像を {style_choice} に変換中..."):
                    processed_image = convert_to_pixel_art(original_image, pixel_size=pixel_size, num_colors=num_colors_pixel)
        
        elif style_choice == "アニメ風":
            st.sidebar.markdown("線画、色、ぼかし、影・ハイライトを調整できます。")
            col1_anime, col2_anime = st.sidebar.columns(2)
            with col1_anime:
                st.subheader("線画/色面")
                line_threshold1 = st.slider("Canny低閾値 (細かい線)", 10, 100, 50)
                line_threshold2 = st.slider("Canny高閾値 (太い線)", 50, 250, 150)
                line_thickness = st.slider("線画の太さ (0=細い, 3=太い)", 0, 3, 1)
                color_clusters = st.slider("色のクラスター数", 2, 64, 16)
                blur_strength = st.slider("色面のぼかし", 0, 10, 2)
            with col2_anime:
                st.subheader("陰影/特殊効果")
                shadow_intensity = st.slider("影の濃さ (0.0=薄い, 1.0=濃い)", 0.0, 1.0, 0.3, 0.05)
                highlight_intensity = st.slider("ハイライトの明るさ (0.0=暗い, 1.0=明るい)", 0.0, 1.0, 0.2, 0.05)
                pencil_texture_intensity = st.slider("色鉛筆テクスチャの強さ", 0.0, 1.0, 0.0, 0.05)
            
            if st.sidebar.button("アニメ風に変換"):
                with st.spinner(f"画像を {style_choice} に変換中..."):
                    processed_image = convert_to_anime_style(original_image, line_threshold1, line_threshold2, line_thickness, color_clusters, blur_strength, shadow_intensity, highlight_intensity, pencil_texture_intensity)
        
        elif style_choice == "スケッチ風":
            if st.sidebar.button("スケッチ風に変換"):
                with st.spinner(f"画像を {style_choice} に変換中..."):
                    processed_image = convert_to_sketch(original_image)
        
        elif style_choice == "油絵風":
            st.sidebar.markdown("フィルタの強さで筆致の荒さが変わります。")
            oil_brush_size = st.sidebar.slider("筆致のサイズ", 2, 20, 5)
            oil_intensity = st.sidebar.slider("色の均一化", 1, 50, 20)
            use_stylization = st.sidebar.checkbox("OpenCVの高度な油絵フィルタを使用", value=True)
            if st.sidebar.button("油絵風に変換"):
                with st.spinner(f"画像を {style_choice} に変換中..."):
                    processed_image = convert_to_oil_painting(original_image, oil_brush_size, oil_intensity, use_stylization)
        else:
            st.sidebar.info("画風スタイルを選択してください。")

    elif transform_type_choice == "エフェクト・ぼかし":
        st.sidebar.subheader("エフェクト・ぼかしの種類を選択")
        effect_choice = st.sidebar.selectbox(
            "適用するエフェクトまたはぼかしを選択してください",
            ("選択してください", "ぼかし", "シャープ", "セピア", "モノクロ", "ネガティブ反転")
        )
        current_style_or_effect_name = effect_choice

        if effect_choice == "ぼかし":
            blur_radius = st.sidebar.slider("ぼかしの強さ", 0.0, 10.0, 2.0, 0.1)
            if st.sidebar.button("ぼかしを適用"):
                with st.spinner(f"画像を {effect_choice} に変換中..."):
                    processed_image = apply_blur(original_image, blur_radius)
        
        elif effect_choice == "シャープ":
            sharpen_factor = st.sidebar.slider("シャープの強さ", 0.5, 5.0, 1.5, 0.1)
            if st.sidebar.button("シャープを適用"):
                with st.spinner(f"画像を {effect_choice} に変換中..."):
                    processed_image = apply_sharpen(original_image, sharpen_factor)
        
        elif effect_choice == "セピア":
            if st.sidebar.button("セピアを適用"):
                with st.spinner(f"画像を {effect_choice} に変換中..."):
                    processed_image = apply_sepia(original_image)
        
        elif effect_choice == "モノクロ":
            if st.sidebar.button("モノクロを適用"):
                with st.spinner(f"画像を {effect_choice} に変換中..."):
                    processed_image = apply_grayscale(original_image)
        
        elif effect_choice == "ネガティブ反転":
            if st.sidebar.button("ネガティブ反転を適用"):
                with st.spinner(f"画像を {effect_choice} に変換中..."):
                    processed_image = apply_invert(original_image)
        else:
            st.sidebar.info("エフェクトの種類を選択してください。")

    elif transform_type_choice == "特定の色を変換/強調":
        st.sidebar.subheader("特定の色域の操作")
        st.sidebar.info("画像から検出された主要な色、または色ピッカーで手動で色を選択してください。")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**画像から検出された主要な色:**")
        
        try:
            dominant_colors_rgb = extract_dominant_colors(original_image, num_colors=8)
            cols = st.sidebar.columns(len(dominant_colors_rgb))
            for i, color_rgb in enumerate(dominant_colors_rgb):
                hex_color = '#%02x%02x%02x' % (color_rgb[0], color_rgb[1], color_rgb[2])
                with cols[i]:
                    if st.button(f'　　　', key=f'color_button_{i}', help=hex_color):
                        update_hsv_from_hex(hex_color)
                    st.sidebar.markdown(f'<div style="width: 100%; height: 20px; background-color: {hex_color}; border: 1px solid #ddd;"></div>', unsafe_allow_html=True)

        except Exception as e:
            st.sidebar.warning(f"主要な色の検出中にエラーが発生しました: {e}")
            st.sidebar.info("画像が小さい、または色のバリエーションが少ない場合に発生することがあります。")

        st.sidebar.markdown("---")
        
        picked_rgb_manual = st.sidebar.color_picker(
            "手動で色を選択",  
            st.session_state.initial_color_hex,  
            key="manual_color_picker"
        )
        
        if picked_rgb_manual != st.session_state.initial_color_hex:
            update_hsv_from_hex(picked_rgb_manual)

        st.sidebar.write(f"**現在選択中のHSV値:** (H: {st.session_state.target_h}, S: {st.session_state.target_s}, V: {st.session_state.target_v})")
        st.sidebar.write("このHSV値を基準に操作が行われます。")

        color_tolerance = st.sidebar.slider(
            "色相(H)の許容範囲 (0-90, 小さいほど厳密)",  
            0, 90, st.session_state.last_color_tolerance
        )
        sat_val_tolerance = st.sidebar.slider(
            "彩度(S)と明度(V)の許容範囲 (0-100, 小さいほど厳密)",  
            0, 100, st.session_state.last_sat_val_tolerance
        )
        
        st.session_state.last_color_tolerance = color_tolerance
        st.session_state.last_sat_val_tolerance = sat_val_tolerance

        # カラーポップの選択肢を削除
        effect_type = st.sidebar.radio(
            "どのように操作しますか？",
            ("選択してください", "彩度強調", "色相変換", "モノクロ化（背景）") 
        )
        current_style_or_effect_name = f"色操作_HSV({st.session_state.target_h},{st.session_state.target_s},{st.session_state.target_v})_{effect_type}"

        new_hue = None
        if effect_type == "色相変換":
            st.sidebar.subheader("変換後の色を選択")
            new_picked_rgb = st.sidebar.color_picker("変換後の色を選択", "#0000FF")
            nr, ng, nb = int(new_picked_rgb[1:3], 16), int(new_picked_rgb[3:5], 16), int(new_picked_rgb[5:7], 16)
            new_rgb_color = np.uint8([[[nb, ng, nr]]])
            new_hsv_color = cv2.cvtColor(new_rgb_color, cv2.COLOR_BGR2HSV)
            new_hue = new_hsv_color[0][0][0]
            st.sidebar.write(f"**変換後の色相(H):** {new_hue}")

        if st.sidebar.button("色操作を実行"):
            if effect_type != "選択してください":
                with st.spinner(f"画像を特定の色を {effect_type} に変換中..."):
                    processed_image = apply_color_manipulation(
                        original_image,  
                        st.session_state.target_h,
                        st.session_state.target_s,
                        st.session_state.target_v,
                        color_tolerance,  
                        sat_val_tolerance,
                        effect_type,
                        new_hue=new_hue
                    )
            else:
                st.sidebar.info("操作したい色と方法を選択してください。")
        
    elif transform_type_choice == "レトロ・ノスタルジック風":
        st.sidebar.subheader("レトロ・ノスタルジック効果")
        retro_effect_choice = st.sidebar.selectbox(
            "適用するレトロ効果を選択してください",
            ("選択してください", "RGBチャンネルずれ", "グリッチ効果")
        )
        current_style_or_effect_name = retro_effect_choice

        if retro_effect_choice == "RGBチャンネルずれ":
            shift_amount = st.sidebar.slider("ずれの量", 1, 20, 5)
            if st.sidebar.button("RGBチャンネルずれを適用"):
                with st.spinner(f"画像を {retro_effect_choice} に変換中..."):
                    processed_image = apply_rgb_channel_shift(original_image, shift_amount)
        elif retro_effect_choice == "グリッチ効果":
            glitch_intensity = st.sidebar.slider("グリッチの強さ", 0.0, 0.5, 0.1, 0.01)
            num_glitches = st.sidebar.slider("グリッチの数", 1, 20, 5)
            if st.sidebar.button("グリッチ効果を適用"):
                with st.spinner(f"画像を {retro_effect_choice} に変換中..."):
                    processed_image = apply_glitch_effect(original_image, intensity=glitch_intensity, num_glitches=num_glitches)
        else:
            st.sidebar.info("レトロ効果の種類を選択してください。")

    elif transform_type_choice == "画像モザイク加工":
        st.sidebar.subheader("画像モザイク効果")
        block_size = st.sidebar.slider("モザイクのブロックサイズ", 2, 100, 10)
        current_style_or_effect_name = transform_type_choice

        if st.sidebar.button("モザイク加工を適用"):
            with st.spinner(f"画像を {transform_type_choice} に変換中..."):
                processed_image = apply_mosaic_effect(original_image, block_size)
    
    elif transform_type_choice == "クールなスタイル": 
        st.sidebar.subheader("クールなスタイルを選択")
        cool_style_choice = st.sidebar.selectbox(
            "適用するクールなスタイルを選択してください",
            ("選択してください", "デュオトーン", "ハーフトーン")
        )
        current_style_or_effect_name = cool_style_choice

        if cool_style_choice == "デュオトーン":
            st.sidebar.markdown("画像を2色調に変換し、スタイリッシュな見た目を演出します。")
            color1 = st.sidebar.color_picker("暗い部分の色", "#000000", key="duotone_color1")
            color2 = st.sidebar.color_picker("明るい部分の色", "#00FFFF", key="duotone_color2")
            duotone_balance = st.sidebar.slider("色のバランス (0.0=暗い色強調, 1.0=明るい色強調)", 0.0, 1.0, 0.5, 0.05)
            
            c1_rgb = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
            c2_rgb = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))

            if st.sidebar.button("デュオトーンを適用"):
                with st.spinner(f"画像を {cool_style_choice} に変換中..."):
                    processed_image = apply_duotone(original_image, color1=c1_rgb, color2=c2_rgb, balance=duotone_balance)

        elif cool_style_choice == "ハーフトーン":
            st.sidebar.markdown("画像を網点（ドット）で表現します。様々なドットの形状、角度、コントラストを試せます。")
            dot_size = st.sidebar.slider("ドットのサイズ", 2, 20, 5)
            dot_shape = st.sidebar.selectbox("ドットの形状", ["circle", "square", "diamond"])
            st.sidebar.info("四角形とひし形の場合のみ、個々のドットの形状が回転します。")
            dot_angle = st.sidebar.slider("ドットの角度 (deg)", 0, 180, 0)
            halftone_contrast = st.sidebar.slider("コントラスト", 0.5, 2.0, 1.0, 0.1)
            if st.sidebar.button("ハーフトーンを適用"):
                with st.spinner(f"画像を {cool_style_choice} に変換中..."):
                    processed_image = apply_halftone(original_image, dot_size=dot_size, shape=dot_shape, angle=dot_angle, contrast_factor=halftone_contrast)

        else:
            st.sidebar.info("クールなスタイルを選択してください。")
            
    else:
        st.sidebar.info("左のサイドバーから変換タイプを選択してください。")

    if processed_image is not None:
        with col_proc:
            st.subheader("加工後の画像")
            st.image(processed_image, use_column_width=True)

        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        download_filename = f"processed_image_{current_style_or_effect_name}.png"
        st.download_button(
            label="加工画像をダウンロード",
            data=byte_im,
            file_name=download_filename,
            mime="image/png"
        )
    else:
        with col_proc:
            st.subheader("加工後の画像")
            st.info("加工オプションを選択し、「適用」ボタンを押してください。")

else:
    st.info("画像をアップロードすると、画風変換、エフェクト適用、特定の色変換、画像加工が楽しめます！")
    st.markdown("---")
    st.subheader("アプリの特長")
    st.markdown("""
    - **AIフィルター提案**: 画像の特性をAIが分析し、最適なフィルターやスタイルを自動で提案します。
    - **画風変換**: ドット絵、アニメ風、スケッチ、油絵など、多様な芸術スタイルに変換します。アニメ風には**色鉛筆テクスチャ**も追加。
    - **特定の色を変換/強調**: AI（K-means）が画像の色を分析し、主要な色を自動で提示。クリック一つでその色域をターゲットに、彩度強調、色相変換、**背景モノクロ化**といった高度な色操作が可能です。
    - **レトロ・ノスタルジック風**: **RGBチャネル分離**や**グリッチエフェクト**で、昔のテレビ画面のようなユニークな視覚効果を付与します。
    - **画像モザイク加工**: 画像全体または一部を、粗いモザイク状に変換し、プライバシー保護や芸術的な表現に活用できます。
    - **クールなスタイル (強化!)**:
        - **デュオトーン**: 画像を2つの指定した色で再着色し、モダンでスタイリッシュな印象を与えます。**色のバランス調整**で、より細かく色合いをコントロールできます。
        - **ハーフトーン**: 画像を網点（ドット）で表現し、コミックやレトロな印刷物のような効果を生成します。**ドットの形状（円、四角、ひし形）、角度、コントラスト**を調整して、多様な表現が可能です。
    """)
