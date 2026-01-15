#main_controller4 only but with face_recognition6

#!/usr/bin/env python3
"""
Mobile-Optimized Main Controller for Alzheimer's App
- Single process (mobile-friendly)
- Progressive loading with user feedback
- Memory efficient
- Optimized for mobile deployment 
"""
import time
global_start_time = time.time()

import os
import warnings
import threading
from queue import Queue

# Mobile-optimized environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #CPU only (mobile doesn't have CUDA)
os.environ['OMP_NUM_THREADS'] = '2'  #Using 2 threads max (mobile CPU)
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

print("Starting Alzheimer's Memory Assistant...")

class MobileProgressTracker:
    """Track loading progress for mobile UI"""
    def __init__(self):
        self.total_steps = 5
        self.current_step = 0
        self.step_names = [
            "Initializing system",
            "Loading speech recognition", 
            "Loading face recognition",
            "Loading AI models",
            "Preparing voice output"
        ]
    
    def update(self, step_name=None):
        self.current_step += 1
        if step_name:
            print(f"Step {self.current_step}/{self.total_steps}: {step_name}")
        else:
            print(f"Step {self.current_step}/{self.total_steps}: {self.step_names[self.current_step-1]}")
        
        # For mobile UI: return progress percentage
        progress = (self.current_step / self.total_steps) * 100
        return progress

class MobileAlzheimerApp:
    def __init__(self):
        self.total_start_time = time.time()
        self.progress = MobileProgressTracker()
        self.components_loaded = False
        self.loading_errors = []
        
        # Core components (loaded progressively)
        self.agent = None
        self.engine = None
        self.recognize_speech = None
        self.recognize_face_from_image = None
        
        # Patient context (customizable per patient)
        self.context_map = {}
        """
        self.context_map = {
            "A": "This is your daughter, A.",
            "B": "This is your grandson, B.",
            "C": "This is your friend, C.",
            "D": "This is your brother, D.",
            "E": "This is your doctor, E.",
            "F": "This is your uncle, F.",
            "G": "This is your son, G.",
            "H": "This is your aunt, H.",
            "I": "This is your neighbour, I.",
            "J": "This is your cousin, J.",
            "K": "This is your granddaughter, K."
        }"""
    
    def load_basic_components(self):
        """Load basic Python components (fast)"""
        try:
            self.progress.update("Setting up basic components")
            
            import asyncio
            import json
            self.asyncio = asyncio
            
            return True
        except Exception as e:
            self.loading_errors.append(f"Basic components: {e}")
            return False
    
    def load_speech_recognition(self):
        """Load speech recognition (medium speed)"""
        try:
            self.progress.update("Loading speech recognition")
            
            from vosk_stt import recognize_speech
            from database.db_utils import find_person_by_name
            self.recognize_speech = recognize_speech
            
            return True
        except Exception as e:
            self.loading_errors.append(f"Speech recognition: {e}")
            return False
    
    def load_face_recognition(self):
        """Load face recognition (slow - show detailed progress)"""
        try:
            self.progress.update("Loading face recognition AI")
            
            print("   Loading face recognition models...")
            face_start = time.time()
            
            from face_recognition6 import recognize_face_from_image  #face_recognition6
            self.recognize_face_from_image = recognize_face_from_image
            
            face_time = time.time() - face_start
            print(f"   Face recognition ready in {face_time:.1f}s")
            
            return True
        except Exception as e:
            self.loading_errors.append(f"Face recognition: {e}")
            return False
    
    def load_ai_models(self):
        """Load RASA AI models"""
        try:
            self.progress.update("Loading conversation AI")
            
            print("   Loading conversation AI...")
            rasa_start = time.time()
            
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            
            from rasa.core.agent import Agent
            self.agent = Agent.load("models\20250727-020507-international-rosemary.tar.gz") #model path
            
            rasa_time = time.time() - rasa_start
            print(f"   Conversation AI ready in {rasa_time:.1f}s")
            
            return True
        except Exception as e:
            self.loading_errors.append(f"AI models: {e}")
            return False
    
    def load_text_to_speech(self):
        """Load text-to-speech (fast)"""
        try:
            self.progress.update("Setting up voice output")
            
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Mobile-optimized TTS settings
            self.engine.setProperty('rate', 150)  # Slower for elderly
            self.engine.setProperty('volume', 0.9)  # Loud and clear
            
            return True
        except Exception as e:
            self.loading_errors.append(f"Text-to-speech: {e}")
            return False
    
    def initialize_app(self):
        """Initialize all components with progress tracking"""
        print(f"Initializing Alzheimer's Memory Assistant...")
        
        loading_steps = [
            self.load_basic_components,
            self.load_speech_recognition,
            self.load_face_recognition,  # This is the slow one
            self.load_ai_models,
            self.load_text_to_speech
        ]
        
        for step_func in loading_steps:
            success = step_func()
            if not success:
                print(f"Loading failed at step: {step_func.__name__}")
                self.show_loading_errors()
                return False
        
        self.components_loaded = True
        total_time = time.time() - self.total_start_time
        
        print(f"\nApp initialized successfully in {total_time:.1f} seconds!")
        print("Ready to assist Alzheimer's patients")
        
        return True
    
    def show_loading_errors(self):
        """Show any loading errors"""
        if self.loading_errors:
            print("\nLoading Errors:")
            for error in self.loading_errors:
                print(f"   - {error}")
    
    def speak_response(self, text):
        """Speak response with mobile-optimized settings"""
        print(f"Speaking: {text}")
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()
    
    def process_intent(self, text):
        """Process user intent"""
        if not self.agent:
            return None, None, None
        
        try:
            result = self.asyncio.run(self.agent.parse_message(text))
            intent = result["intent"]["name"]
            confidence = result["intent"]["confidence"]
            entity_value = result["entities"][0]["value"] if result["entities"] else None
            
            print(f"Intent: {intent} (confidence: {confidence:.3f})")
            if entity_value:
                print(f"Entity: {entity_value}")
            
            return intent, confidence, entity_value
        except Exception as e:
            print(f"Intent processing failed: {e}")
            return None, None, None
    
    def recognize_face(self, image_path):
        """Recognize face from image"""
        if not self.recognize_face_from_image:
            return None, 0.0
        
        try:
            print("Analyzing face...")
            start_time = time.time()
            
            best_match, score = self.recognize_face_from_image(image_path)
            
            processing_time = time.time() - start_time
            print(f"Face analysis completed in {processing_time:.1f}s")
            print(f" Recognized: {best_match} (confidence: {score:.4f})")
            
            return best_match, score
        except Exception as e:
            print(f"Face recognition failed: {e}")
            return None, 0.
        
    def get_person_description(self, name: str) -> str:
        """Get relationship description from database"""
        from database.db_utils import find_person_by_name
        record = find_person_by_name(name)
        if record:
            _, db_name, db_relation, *_ = record
            return f"This is your {db_relation}, {db_name}."
        return None

    
    def run_conversation_loop(self):
        """Main conversation loop optimized for elderly users"""
        if not self.components_loaded:
            print("App not properly initialized!")
            return
        
        print("\nStarting conversation mode...")
        print("Listening for your voice or you can type...")
        
        best_match = None
        
        while True:
            try:
                # Get user input (voice or text)
                print("\n" + "="*50)
                print("Please speak now, or press Enter to type...")
                
                if self.recognize_speech:
                    text = self.recognize_speech()
                    if not text.strip():
                        text = input("Type your message: ").strip()
                else:
                    text = input("Type your message: ").strip()
                
                if not text:
                    continue
                
                print(f"You said: {text}")
                
                # Process intent
                intent, confidence, entity_value = self.process_intent(text)
                
                # Generate response
                response_text = "I'm not sure how to help with that."
                
                if intent == "greet":
                    response_text = "Hello! How can I assist you today?"
                
                elif intent == "goodbye":
                    response_text = "Goodbye! Take care"
                    self.speak_response(response_text)
                    break
                
                elif intent == "bot_challenge":
                    response_text = "I am your memory assistant. "
                
                elif intent == "ask_identity":
                    print("\nPlease provide an image...")
                    image_path = input("Enter image path: ").strip().strip('"')

                    if image_path:
                        best_match, score = self.recognize_face(image_path)

                        if best_match:
                            desc = self.get_person_description(best_match)
                            response_text = desc if desc else "I'm sorry, I don't recognize this person clearly."
                        else:
                            response_text = "I'm sorry, I couldn't detect any recognizable person."
                    else:
                        response_text = "No photo provided. Please try again with a photo."


                elif intent == "verify_person":
                    if best_match:
                        actual_desc = self.get_person_description(best_match)
                        if actual_desc:
                            if entity_value and entity_value.lower() == best_match.lower():
                                response_text = f"Yes, that's correct. {actual_desc}"
                            else:
                                response_text = f"No, actually {actual_desc}"
                        else:
                            response_text = "I recognize the person but don't have enough details."
                    else:
                        response_text = "I need to see a photo first before I can verify who someone is."

                
                # Speak response
                self.speak_response(response_text)
                
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                self.speak_response("I'm sorry, something went wrong. Please try again.")

def main():
    """Main function for mobile app"""
    app = MobileAlzheimerApp()
    
    # Initialize app with progress tracking
    if app.initialize_app():
        total_app_ready_time = time.time() - global_start_time
        print(f"\nTotal time (env + imports + model loading): {total_app_ready_time:.2f} seconds")

        # Run main conversation loop
        app.run_conversation_loop()
    else:
        print("App failed to initialize properly")
        app.show_loading_errors()

if __name__ == "__main__":
    main()