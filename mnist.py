import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from PIL import Image
import io
import sys

# Handle missing matplotlib gracefully
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.error("matplotlib not found. Install it with: pip install matplotlib")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def load_mnist_data():
    """Load and preprocess MNIST dataset - cached for performance"""
    with st.spinner("Loading MNIST dataset... (this may take a moment)"):
        try:
            # Load MNIST dataset
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            # Normalize pixel values to [0, 1]
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Reshape data to add channel dimension
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            
            # Convert labels to categorical
            y_train = keras.utils.to_categorical(y_train, 10)
            y_test = keras.utils.to_categorical(y_test, 10)
            
            return (x_train, y_train), (x_test, y_test)
        except Exception as e:
            st.error(f"Error loading MNIST data: {e}")
            st.stop()

def create_lightweight_cnn():
    """Create a lighter CNN model for faster training"""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model_fast(model, x_train, y_train, x_test, y_test, epochs=5):
    """Train model with progress bar"""
    # Use smaller batch size and fewer epochs for faster training
    batch_size = 128
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom callback to update progress
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch + 1}/{epochs} - Accuracy: {logs["accuracy"]:.4f} - Val Accuracy: {logs["val_accuracy"]:.4f}')
    
    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[ProgressCallback()],
        verbose=0
    )
    
    progress_bar.progress(1.0)
    status_text.text("Training completed!")
    
    return history

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 28x28
        img_array = cv2.resize(img_array, (28, 28))
        
        # Invert if needed (MNIST digits are white on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def plot_simple_chart(data, title, xlabel, ylabel):
    """Simple plotting function using matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        st.error("Matplotlib not available for plotting")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(data) + 1)
    ax.plot(epochs, data, 'b-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def main():
    st.title("🔢 MNIST Digit Classifier (Optimized)")
    st.markdown("Fast CNN training and prediction for handwritten digits")
    
    # Check dependencies
    st.sidebar.header("🚀 Quick Start")
    st.sidebar.info("1. Load Dataset\n2. Train Model (5-10 epochs)\n3. Test Predictions")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dataset & Training", "🔮 Prediction", "ℹ️ About"])
    
    with tab1:
        st.header("Dataset & Model Training")
        
        # Load data button
        if not st.session_state.data_loaded:
            if st.button("🔄 Load MNIST Dataset", type="primary"):
                (x_train, y_train), (x_test, y_test) = load_mnist_data()
                st.session_state.train_data = (x_train, y_train)
                st.session_state.test_data = (x_test, y_test)
                st.session_state.data_loaded = True
                st.success("✅ Dataset loaded successfully!")
                st.rerun()
        
        if st.session_state.data_loaded:
            # Display dataset info
            x_train, y_train = st.session_state.train_data
            x_test, y_test = st.session_state.test_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", f"{len(x_train):,}")
            with col2:
                st.metric("Test Samples", f"{len(x_test):,}")
            with col3:
                st.metric("Image Size", "28×28")
            with col4:
                st.metric("Classes", "10")
            
            # Show sample images
            if MATPLOTLIB_AVAILABLE:
                st.subheader("Sample Images from Dataset")
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i in range(10):
                    row, col = i // 5, i % 5
                    # Find first occurrence of digit i
                    idx = np.where(np.argmax(y_train, axis=1) == i)[0][0]
                    axes[row, col].imshow(x_train[idx].reshape(28, 28), cmap='gray')
                    axes[row, col].set_title(f'Digit {i}')
                    axes[row, col].axis('off')
                plt.tight_layout()
                st.pyplot(fig)
            
            st.divider()
            
            # Training section
            st.subheader("🏋️ Train CNN Model")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                epochs = st.slider("Training Epochs", min_value=3, max_value=15, value=5, 
                                 help="More epochs = better accuracy but slower training")
            with col2:
                st.info(f"Estimated time: ~{epochs * 10} seconds")
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Creating model..."):
                    # Create model
                    model = create_lightweight_cnn()
                    
                    st.subheader("Model Architecture")
                    # Display model summary in a more compact way
                    total_params = model.count_params()
                    st.write(f"**Total Parameters:** {total_params:,}")
                    
                    # Show layers info
                    layers_info = []
                    for layer in model.layers:
                        if hasattr(layer, 'output_shape'):
                            layers_info.append(f"{layer.name}: {layer.output_shape}")
                    st.write("**Layers:** " + " → ".join([info.split(':')[0] for info in layers_info[:5]]))
                
                # Train model
                st.subheader("Training Progress")
                history = train_model_fast(model, x_train, y_train, x_test, y_test, epochs)
                
                # Store model and results
                st.session_state.model = model
                st.session_state.history = history
                st.session_state.model_trained = True
                
                # Evaluate model
                test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Test Accuracy", f"{test_accuracy:.3f}")
                with col2:
                    st.metric("Final Test Loss", f"{test_loss:.3f}")
                
                # Plot training curves
                if MATPLOTLIB_AVAILABLE:
                    st.subheader("Training Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = plot_simple_chart(history.history['accuracy'], 
                                               'Training Accuracy', 'Epoch', 'Accuracy')
                        if fig1:
                            st.pyplot(fig1)
                    
                    with col2:
                        fig2 = plot_simple_chart(history.history['loss'], 
                                               'Training Loss', 'Epoch', 'Loss')
                        if fig2:
                            st.pyplot(fig2)
                
                st.success("🎉 Model trained successfully! Go to Prediction tab to test it.")
    
    with tab2:
        st.header("🔮 Digit Prediction")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Dataset & Training' tab!")
            return
        
        model = st.session_state.model
        
        # Prediction methods
        prediction_method = st.radio(
            "Choose prediction method:",
            ["📁 Upload Image", "🎲 Random Test Sample"],
            horizontal=True
        )
        
        if prediction_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an image of a handwritten digit", 
                type=['png', 'jpg', 'jpeg'],
                help="Best results with black digit on white background"
            )
            
            if uploaded_file is not None:
                # Display and process image
                image = Image.open(uploaded_file)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, width=200)
                
                with col2:
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        st.subheader("Processed (28×28)")
                        st.image(processed_image.reshape(28, 28), width=200, clamp=True)
                
                with col3:
                    if processed_image is not None:
                        # Make prediction
                        prediction = model.predict(processed_image, verbose=0)
                        predicted_digit = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        st.subheader("Prediction")
                        st.markdown(f"## **{predicted_digit}**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Show top 3 predictions
                        st.subheader("Top 3 Predictions")
                        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                        for i, idx in enumerate(top_3_indices):
                            st.write(f"{i+1}. Digit **{idx}**: {prediction[0][idx]:.1%}")
        
        elif prediction_method == "🎲 Random Test Sample":
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("🎲 Get Random Sample", type="secondary"):
                    x_test, y_test = st.session_state.test_data
                    
                    # Select random sample
                    idx = np.random.randint(0, len(x_test))
                    test_image = x_test[idx]
                    true_label = np.argmax(y_test[idx])
                    
                    # Store in session state
                    st.session_state.random_image = test_image
                    st.session_state.random_label = true_label
            
            if hasattr(st.session_state, 'random_image'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Test Image")
                    st.image(st.session_state.random_image.reshape(28, 28), 
                            width=200, clamp=True)
                    st.write(f"**True Label:** {st.session_state.random_label}")
                
                with col2:
                    # Make prediction
                    prediction = model.predict(st.session_state.random_image.reshape(1, 28, 28, 1), verbose=0)
                    predicted_digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    st.subheader("Model Prediction")
                    st.markdown(f"## **{predicted_digit}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Show result
                    if predicted_digit == st.session_state.random_label:
                        st.success("✅ Correct Prediction!")
                    else:
                        st.error("❌ Incorrect Prediction")
    
    with tab3:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🧠 CNN Architecture
        - **Input Layer:** 28×28×1 (grayscale images)
        - **Conv2D Layer 1:** 16 filters, 5×5 kernel + MaxPooling
        - **Conv2D Layer 2:** 32 filters, 3×3 kernel + MaxPooling
        - **Dense Layer:** 32 neurons + Dropout
        - **Output Layer:** 10 neurons (digits 0-9)
        
        ### 🚀 Performance Optimizations
        - Lightweight model architecture
        - Cached data loading
        - Efficient batch processing
        - Progress tracking during training
        
        ### 📦 Required Dependencies
        ```bash
        pip install streamlit tensorflow opencv-python pillow matplotlib numpy
        ```
        
        ### 💡 Tips for Best Results
        - Use clear, well-centered digit images
        - Black digits on white background work best
        - Train for 5-10 epochs for good accuracy
        - Larger images are automatically resized to 28×28
        """)
        
        if st.session_state.model_trained:
            st.subheader("📊 Current Model Stats")
            model = st.session_state.model
            st.write(f"**Total Parameters:** {model.count_params():,}")
            if hasattr(st.session_state, 'history'):
                final_acc = st.session_state.history.history['val_accuracy'][-1]
                st.write(f"**Final Validation Accuracy:** {final_acc:.3f}")

if __name__ == "__main__":
    main()