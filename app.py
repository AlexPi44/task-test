import streamlit as st
import requests
import sqlite3
import os
import base64
from datetime import datetime
import json
import time
import threading
import faiss  # Added for similarity search
import numpy as np  # Added for array operations
import torch  # Added for embeddings
from transformers import AutoTokenizer, AutoModel  # Added for embeddings

# Set page configuration
st.set_page_config(
    page_title="AI Creative Studio",
    page_icon="🚀",
    layout="wide"
)

# Database setup
DB_PATH = "generations.db"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    enhanced_prompt TEXT NOT NULL,
    image_path TEXT,
    three_d_model_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# Define API endpoint
API_URL = "http://localhost:8888/execution"

# Initialize FAISS index and embedding model
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    dimension = 384  # Dimension of MiniLM embeddings
    index = faiss.IndexFlatL2(dimension)
    prompt_embeddings = {}  # Store prompt embeddings by generation ID
except Exception as e:
    st.error(f"Failed to initialize FAISS or embedding model: {e}")
    tokenizer, embedder, index, prompt_embeddings = None, None, None, {}

def get_embedding(text):
    """Generate embedding for a text prompt."""
    if not tokenizer or not embedder:
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to get file as base64 for download
def get_file_as_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        contents = f.read()
    return base64.b64encode(contents).decode("utf-8")

# Function to display 3D model using Three.js
def display_3d_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"3D model file not found at: {model_path}")
        return
    html_content = f"""
    <html>
      <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js"></script>
        <style>
          body {{ margin: 0; overflow: hidden; }}
          canvas {{ width: 100%; height: 100%; }}
        </style>
      </head>
      <body>
        <div id="container" style="width: 100%; height: 400px;"></div>
        <script>
          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0x222222);
          const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
          camera.position.z = 5;
          const renderer = new THREE.WebGLRenderer({{ antialias: true }});
          renderer.setSize(window.innerWidth, window.innerHeight);
          document.getElementById('container').appendChild(renderer.domElement);
          const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1);
          light.position.set(0, 20, 0);
          scene.add(light);
          const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
          directionalLight.position.set(0, 20, 10);
          scene.add(directionalLight);
          const controls = new THREE.OrbitControls(camera, renderer.domElement);
          controls.update();
          const loader = new THREE.GLTFLoader();
          const modelUrl = "{model_path}";
          loader.load(
            modelUrl,
            function(gltf) {{
              scene.add(gltf.scene);
              const box = new THREE.Box3().setFromObject(gltf.scene);
              const center = box.getCenter(new THREE.Vector3());
              const size = box.getSize(new THREE.Vector3());
              const maxDim = Math.max(size.x, size.y, size.z);
              const scale = 1 / maxDim * 2;
              gltf.scene.scale.set(scale, scale, scale);
              gltf.scene.position.x = -center.x * scale;
              gltf.scene.position.y = -center.y * scale;
              gltf.scene.position.z = -center.z * scale;
            }},
            undefined,
            function(error) {{
              console.error('Error loading model:', error);
            }}
          );
          function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
          }}
          animate();
          window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
          }});
        </script>
      </body>
    </html>
    """
    st.components.v1.html(html_content, height=400)

# Main heading
st.title("🚀 AI Creative Studio")
st.subheader("Turn your ideas into 3D models")

# Sidebar for app information
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=AI+Studio", width=150)
    st.header("About")
    st.write("""
    This app uses advanced AI to transform text prompts into 3D models.
    
    * Enter your prompt
    * DeepSeek AI enhances it
    * Generate stunning images
    * Convert to 3D models
    * Browse your creations
    """)
    st.header("Memory")
    memory_count = cursor.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    st.write(f"📊 {memory_count} creations stored in memory")

# Create tabs for different functions
tab1, tab2 = st.tabs(["Create New", "View Past Creations"])

# Tab 1: Create new generation
with tab1:
    st.header("Create a New Generation")
    
    # Initialize session state for recent prompts
    if "recent_prompts" not in st.session_state:
        st.session_state.recent_prompts = []
    
    # Show recent prompts
    if st.session_state.recent_prompts:
        st.subheader("Recent Prompts")
        selected_prompt = st.selectbox("Select a previous prompt to remix:", [""] + st.session_state.recent_prompts)
        if selected_prompt:
            user_prompt = st.text_area("Enter your creative prompt:", value=selected_prompt, height=100)
        else:
            user_prompt = st.text_area("Enter your creative prompt:", 
                                      placeholder="Example: Make me a glowing dragon standing on a cliff at sunset.",
                                      height=100)
    else:
        user_prompt = st.text_area("Enter your creative prompt:", 
                                  placeholder="Example: Make me a glowing dragon standing on a cliff at sunset.",
                                  height=100)
    
    temperature = st.slider("Creativity Level (Temperature)", 0.1, 1.0, 0.7, 0.1)
    generate_button = st.button("🔮 Generate!", use_container_width=True)
    
    if generate_button and user_prompt:
        with st.spinner("🧠 Enhancing prompt with AI..."):
            # Add prompt to recent prompts (limit to 5)
            if user_prompt not in st.session_state.recent_prompts:
                st.session_state.recent_prompts = [user_prompt] + st.session_state.recent_prompts[:4]
            try:
                payload = {"prompt": user_prompt, "temperature": temperature}
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Generation complete!")
                    if "image_path" in data and os.path.exists(data["image_path"]):
                        st.subheader("Generated Image")
                        st.image(data["image_path"], use_column_width=True)
                        if "three_d_model_path" in data and os.path.exists(data["three_d_model_path"]):
                            st.subheader("3D Model")
                            display_3d_model(data["three_d_model_path"])
                            model_file = get_file_as_base64(data["three_d_model_path"])
                            if model_file:
                                st.download_button(
                                    label="📥 Download 3D Model (GLB)",
                                    data=model_file,
                                    file_name=os.path.basename(data["three_d_model_path"]),
                                    mime="model/gltf-binary"
                                )
                    else:
                        st.error("Image file not found")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the Openfabric server is running on port 8888")

# Tab 2: View past creations
with tab2:
    st.header("Your Past Creations")
    
    # Retrieve past generations
    cursor.execute("SELECT id, prompt, enhanced_prompt, image_path, three_d_model_path, timestamp FROM generations ORDER BY timestamp DESC")
    generations = cursor.fetchall()
    
    # Rebuild FAISS index
    if index and prompt_embeddings is not None:
        index.reset()
        prompt_embeddings.clear()
        cursor.execute("SELECT id, prompt FROM generations")
        for gen_id, prompt in cursor.fetchall():
            embedding = get_embedding(prompt)
            if embedding is not None:
                index.add(embedding)
                prompt_embeddings[gen_id] = prompt
    
    # Similarity search
    if index and tokenizer and embedder:
        st.subheader("Find Similar Creations")
        search_prompt = st.text_input("Enter a prompt to find similar creations:")
        if search_prompt:
            search_embedding = get_embedding(search_prompt)
            if search_embedding is not None:
                distances, indices = index.search(search_embedding, k=3)
                st.write("Similar Creations:")
                for idx in indices[0]:
                    gen_id = list(prompt_embeddings.keys())[idx]
                    prompt = prompt_embeddings[gen_id]
                    st.write(f"Creation #{gen_id}: {prompt}")
    
    # Gallery view
    st.subheader("Gallery")
    cols = st.columns(3)
    for i, gen in enumerate(generations):
        gen_id, prompt, enhanced_prompt, image_path, model_path, timestamp = gen
        with cols[i % 3]:
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption=f"Creation #{gen_id}", width=150)
                if st.button(f"View Details (#{gen_id})", key=f"details_{gen_id}"):
                    st.session_state[f"show_details_{gen_id}"] = True
    
    # Detailed view
    if len(generations) == 0:
        st.info("You haven't created anything yet. Go to the Create New tab to get started!")
    else:
        st.subheader("Detailed View")
        for gen in generations:
            gen_id, prompt, enhanced_prompt, image_path, model_path, timestamp = gen
            try:
                timestamp_formatted = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_formatted = timestamp
            if f"show_details_{gen_id}" in st.session_state and st.session_state[f"show_details_{gen_id}"]:
                with st.expander(f"Creation #{gen_id} - {timestamp_formatted}"):
                    cols = st.columns([1, 2, 1])
                    with cols[0]:
                        st.write(f"**Original Prompt:**\n{prompt}")
                        with st.expander("Enhanced Prompt"):
                            st.write(enhanced_prompt)
                    with cols[1]:
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, use_column_width=True)
                        else:
                            st.error("Image file not found")
                    with cols[2]:
                        if model_path and os.path.exists(model_path):
                            model_file = get_file_as_base64(model_path)
                            if model_file:
                                st.download_button(
                                    label="📥 Download 3D Model",
                                    data=model_file,
                                    file_name=os.path.basename(model_path),
                                    mime="model/gltf-binary"
                                )
                            if st.button(f"👁️ View in 3D (#{gen_id})", key=f"view_3d_{gen_id}"):
                                st.session_state[f"show_3d_{gen_id}"] = True
                            if f"show_3d_{gen_id}" in st.session_state and st.session_state[f"show_3d_{gen_id}"]:
                                display_3d_model(model_path)
                                if st.button(f"Hide 3D Model (#{gen_id})", key=f"hide_3d_{gen_id}"):
                                    st.session_state[f"show_3d_{gen_id}"] = False
                        else:
                            st.error("3D model file not found")

# Footer
st.markdown("---")
st.markdown("Made for the AI Developer Challenge")