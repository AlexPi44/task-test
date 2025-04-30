import logging
import os
import sqlite3
import uuid
from typing import Dict
from datetime import datetime

import torch
from transformers import pipeline

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub

# Initialize the LLM model (only once when the app starts)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
generator = pipeline("text-generation", model="deepseek-ai/deepseek-llm-7b-chat", device=device)
logging.info("Model loaded successfully")

# Set up database for memory storage
DB_PATH = "generations.db"
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

# Create outputs directory if it doesn't exist
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """
    try:
        # Retrieve input
        request: InputClass = model.request
        user_prompt = request.prompt
        if not user_prompt or len(user_prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        logging.info(f"Received prompt: {user_prompt}")

        # Retrieve user config
        user_config: ConfigClass = configurations.get('super-user', None)
        logging.info(f"Configurations: {configurations}")

        # Initialize the Stub with app IDs
        app_ids = user_config.app_ids if user_config else [
            os.getenv("TEXT_TO_IMAGE_APP_ID", "f0997a01-d6d3-a5fe-53d8-561300318557"),
            os.getenv("IMAGE_TO_3D_APP_ID", "69543f29-4d41-4afc-7f29-3d51591f11eb")
        ]
        stub = Stub(app_ids)

        # ------------------------------
        # Use DeepSeek model to enhance the prompt
        # ------------------------------
        system_message = "You are an AI that expands user prompts into detailed descriptions for image generation. Create vivid, descriptive text that will help generate beautiful images. Focus on details, lighting, mood, and composition."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        logging.info("Generating enhanced prompt with DeepSeek model...")
        output = generator(messages, max_new_tokens=200, temperature=0.7)
        
        # Robust extraction
        enhanced_prompt = None
        if isinstance(output, list) and len(output) > 0:
            try:
                enhanced_prompt = output[0]['generated_text'][-1]['content']
            except (IndexError, KeyError, TypeError):
                logging.warning("Falling back to raw output parsing")
                enhanced_prompt = str(output[0]['generated_text'])
        
        if not enhanced_prompt or len(enhanced_prompt.strip()) == 0:
            raise ValueError("Failed to generate enhanced prompt")
        
        logging.info(f"Enhanced prompt: {enhanced_prompt}")
        
        # ------------------------------
        # Generate image from the enhanced prompt
        # ------------------------------
        logging.info("Generating image from enhanced prompt...")
        
        # Call the Text to Image app
        text_to_image_app_id = app_ids[0]
        try:
            image_response = stub.call(
                text_to_image_app_id, 
                {'prompt': enhanced_prompt}, 
                'super-user'
            )
            
            image_output = image_response.get('result')
            if not image_output:
                raise ValueError("No image data received from Text to Image app")
                
            logging.info("Image generated successfully")
            
            # ------------------------------
            # Convert image to 3D model
            # ------------------------------
            logging.info("Converting image to 3D model...")
            
            # Call the Image to 3D app
            image_to_3d_app_id = app_ids[1]
            three_d_response = stub.call(
                image_to_3d_app_id, 
                {'image': image_output}, 
                'super-user'
            )
            
            three_d_output = three_d_response.get('result')
            if not three_d_output:
                raise ValueError("No 3D model data received from Image to 3D app")
                
            logging.info("3D model generated successfully")
            
            # ------------------------------
            # Store in memory (database)
            # ------------------------------
            logging.info("Storing generation in database...")
            
            # Insert into database first to get an ID
            cursor.execute(
                "INSERT INTO generations (prompt, enhanced_prompt) VALUES (?, ?)",
                (user_prompt, enhanced_prompt)
            )
            conn.commit()
            generation_id = cursor.lastrowid
            
            # Save files with the generation ID
            image_filename = f"image_{generation_id}.png"
            image_path = os.path.join(OUTPUTS_DIR, image_filename)
            
            three_d_filename = f"model_{generation_id}.glb"
            three_d_path = os.path.join(OUTPUTS_DIR, three_d_filename)
            
            # Save the image and 3D model
            with open(image_path, "wb") as f:
                f.write(image_output)
                
            with open(three_d_path, "wb") as f:
                f.write(three_d_output)
                
            # Update database with file paths
            cursor.execute(
                "UPDATE generations SET image_path = ?, three_d_model_path = ? WHERE id = ?",
                (image_path, three_d_path, generation_id)
            )
            conn.commit()
            
            logging.info(f"Generation saved with ID: {generation_id}")
            
            # ------------------------------
            # Prepare response
            # ------------------------------
            response: OutputClass = model.response
            response.message = f"3D model generated successfully for prompt: {user_prompt}"
            
            # Set additional fields for the Streamlit app
            try:
                response.generation_id = generation_id
                response.image_path = image_path
                response.three_d_model_path = three_d_path
            except AttributeError:
                logging.warning("OutputClass does not support additional fields. Only message will be returned.")
            
        except Exception as e:
            logging.error(f"Error during generation process: {str(e)}")
            response: OutputClass = model.response
            response.message = f"Error generating 3D model: {str(e)}"
            
    except Exception as e:
        logging.error(f"Unhandled exception in execute: {str(e)}")
        response: OutputClass = model.response
        response.message = f"Internal server error: {str(e)}"