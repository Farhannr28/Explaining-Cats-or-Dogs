import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
from io import BytesIO
import os
import cv2

def build_convnext_model(input_shape=(384, 384, 3)):
    """Build the ConvNeXt model architecture from saved .h5 file"""
    base_model = ConvNeXtBase(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=False
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="ConvNeXt_Binary")
    return model

def preprocess_image(img_path, target_size=(384, 384)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def make_gradcam_heatmap(img_array, model):
    """
    Robust Manual Grad-CAM that avoids Graph Surgery
    """
    # Separate the Feature Extractor from the Head (Classifier)
    base_model_layer = model.layers[1]
    
    # Extract Features
    conv_output = base_model_layer(img_array, training=False)
    
    # Define the Classifier Head manually using the remaining layers
    classifier_layers = model.layers[2:]
    
    def classifier_forward(features):
        x = features
        for layer in classifier_layers:
            x = layer(x, training=False)
        return x

    # Calculate Gradients
    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        
        preds = classifier_forward(conv_output)

        top_class_channel = preds[:, 0]

    # Get Gradient of the Output w.r.t the Feature Maps
    grads = tape.gradient(top_class_channel, conv_output)

    # Global Average Pooling of Gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted Sum
    conv_output = conv_output[0] 
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def explain_with_gradcam(model, img_array, save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)

    # Get Prediction
    preds = model.predict(img_array, verbose=0)
    pred_score = preds[0][0]
    predicted_class = 1 if pred_score > 0.5 else 0
    confidence = pred_score if predicted_class == 1 else (1 - pred_score)
    
    print(f"Prediction: {'Dog' if predicted_class==1 else 'Cat'} ({confidence:.2%})")

    # Generate Heatmap using the ROBUST method
    print("Generating Grad-CAM heatmap...")
    heatmap = make_gradcam_heatmap(img_array, model)

    # Visualization
    original_img = (img_array[0] * 255).astype(np.uint8)
    heatmap = np.uint8(255 * heatmap)
    
    # Resize to image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Colorize
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # Save
    pil_img = Image.fromarray(superimposed_img)
    file_path = os.path.join(save_dir, 'gradcam_explanation.png')
    pil_img.save(file_path)

    # Base64 Encoder
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        'base64': f"data:image/png;base64,{img_base64}",
        'file_path': file_path,
        'prediction': predicted_class,
        'confidence': float(confidence),
        'raw_prediction': float(pred_score)
    }

def main():
    MODEL_WEIGHTS_PATH = 'convnext_model.h5' 
    IMAGE_PATH = 'test/t.jpg'
    CLASS_LABELS = ['Cat', 'Dog']
    INPUT_SHAPE = (384, 384, 3)
    
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights not found at {MODEL_WEIGHTS_PATH}")
        return
        
    print("Building model...")
    model = build_convnext_model(input_shape=INPUT_SHAPE)
    
    print(f"Loading weights...")
    model.load_weights(MODEL_WEIGHTS_PATH)
    
    print(f"Loading image...")
    img_array = preprocess_image(IMAGE_PATH, target_size=INPUT_SHAPE[:2])
    
    print("\nRunning Robust Grad-CAM...")
    result = explain_with_gradcam(model, img_array)
    
    print("\nRESULTS")
    print(f"Saved to: {result['file_path']}")
    print(f"Prediction: {CLASS_LABELS[result['prediction']]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()