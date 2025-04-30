AI Creative Studio
This project is a submission for the AI Developer Challenge, creating an end-to-end pipeline that transforms user prompts into 3D models using a local LLM (DeepSeek), Openfabric apps, and a Streamlit GUI.

Features
Prompt Enhancement: Uses DeepSeek to expand user prompts into vivid descriptions, with adjustable creativity (temperature).
Text-to-Image: Calls an Openfabric app to generate images.
Image-to-3D: Converts images into 3D models using another Openfabric app.


Memory:
Long-Term: Stores prompts, images, and 3D models in SQLite.
Short-Term: Remembers recent prompts in the Streamlit session for remixing.
GUI: Streamlit interface with 3D visualization, gallery view, and similarity search.
Robustness: Retry logic for API calls, robust error handling, and detailed logging.
Bonus: FAISS-based similarity search for finding similar creations.


🛠 Setup
Prerequisites
Python 3.8+
Docker (optional)
NVIDIA GPU (recommended for DeepSeek)
Poetry for dependency management
Local Installation
Clone the repository:
git clone <repo-url>
cd ai-test
