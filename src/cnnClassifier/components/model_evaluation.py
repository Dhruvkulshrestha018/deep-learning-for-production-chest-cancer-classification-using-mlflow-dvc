import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
import os

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        # Check if directory exists
        if not Path(self.config.training_data).exists():
            raise FileNotFoundError(f"Training data directory not found: {self.config.training_data}")
            
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
    
    def evaluation(self):
        # Check if model exists
        if not Path(self.config.path_of_model).exists():
            raise FileNotFoundError(f"Model not found: {self.config.path_of_model}")
            
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        self.log_into_mlflow()
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
    
        try:
            # DO NOT hardcode credentials here - they're already in environment variables
            # Just set the tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_registry_uri(self.config.mlflow_uri)
            
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Registry URI: {mlflow.get_registry_uri()}")
            
            # Check if credentials are set (for debugging)
            print(f"Username set: {'MLFLOW_TRACKING_USERNAME' in os.environ}")
            print(f"Password set: {'MLFLOW_TRACKING_PASSWORD' in os.environ}")
            
            with mlflow.start_run():
                # Log parameters and metrics
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics({
                    "loss": self.score[0],
                    "accuracy": self.score[1]
                })
                
                # Log model
                mlflow.tensorflow.log_model(
                    model=self.model,
                    artifact_path="model",
                    registered_model_name="VGG16Model"
                )
                
                print(f"Run ID: {mlflow.active_run().info.run_id}")
                print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")
                
        except Exception as e:
            print(f"MLflow logging failed: {e}")
            raise e