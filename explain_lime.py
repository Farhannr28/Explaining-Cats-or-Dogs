# explain_lime.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import base64
from io import BytesIO
import os

def build_convnext_model(input_shape=(384, 384, 3)):
    """Build the ConvNeXt model architecture"""
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
    """Load and preprocess image for model input"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def explain_with_lime(model, img_array, save_dir='output'):
    """
    Generate LIME explanation and return base64 image
    
    Returns:
        dict with base64 image, file path, prediction, and confidence
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # LIME requires a prediction function that returns probabilities for both classes
    def predict_fn(images):
        preds = model.predict(images, verbose=0)
        # Convert binary output to 2-class probabilities
        # preds is shape (n, 1) with values [0, 1]
        # We need shape (n, 2) where [:, 0] is prob of class 0, [:, 1] is prob of class 1
        return np.hstack([1 - preds, preds])
    
    print("Creating LIME explainer...")
    explainer = lime_image.LimeImageExplainer()
    
    print("Generating explanation (this may take 1-2 minutes)...")
    explanation = explainer.explain_instance(
        img_array[0].astype('double'), 
        predict_fn, 
        top_labels=2,  # Explain both classes
        hide_color=0, 
        num_samples=1000
    )
    
    # Get the predicted class (0 or 1)
    pred = model.predict(img_array, verbose=0)[0][0]
    predicted_class = 1 if pred > 0.5 else 0
    confidence = pred if predicted_class == 1 else (1 - pred)
    
    # Get explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        predicted_class, 
        positive_only=True, 
        num_features=5, 
        hide_rest=True
    )
    
    # Generate explanation image
    print("Creating explanation image...")
    explanation_numpy = mark_boundaries(temp / 2 + 0.5, mask)
    explanation_uint8 = (explanation_numpy * 255).astype(np.uint8)
    pil_img = Image.fromarray(explanation_uint8)
    
    # Save to file
    file_path = os.path.join(save_dir, 'lime_explanation.png')
    pil_img.save(file_path)
    
    # Generate base64
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        'base64': f"data:image/png;base64,{img_base64}",
        'file_path': file_path,
        'prediction': predicted_class,
        'confidence': float(confidence),
        'raw_prediction': float(pred)
    }

def main():
    # Configuration
    MODEL_WEIGHTS_PATH = 'convnext_model.h5'
    IMAGE_PATH = 'Test_Akmal.png'
    CLASS_LABELS = ['Cat', 'Dog']  # Update with your actual class names
    INPUT_SHAPE = (384, 384, 3)
    
    # Check if files exist
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
        exit(1)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'")
        print(f"Current directory: {os.getcwd()}")
        print("\nAvailable image files:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {file}")
        exit(1)
    
    print("Building model architecture...")
    model = build_convnext_model(input_shape=INPUT_SHAPE)
    
    print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("✓ Model loaded successfully!")
    
    print(f"\nLoading image from: {IMAGE_PATH}")
    img_array = preprocess_image(IMAGE_PATH, target_size=INPUT_SHAPE[:2])
    print("✓ Image loaded successfully!")
    
    print("\n" + "="*60)
    print("Generating LIME explanation...")
    print("="*60)
    
    result = explain_with_lime(model, img_array)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Explanation saved to: {result['file_path']}")
    print(f"  Predicted class: {result['prediction']} ({CLASS_LABELS[result['prediction']]})")
    print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"  Raw prediction: {result['raw_prediction']:.4f}")
    print("="*60)
    
    return result

if __name__ == "__main__":
    try:
        result = main()
        print("\n✓ Successfully generated LIME explanation!")
        print(f"Base64 string length: {len(result['base64'])} characters")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()