from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier import config, logger


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        
        print(f"Model path: {eval_config.path_of_model}")
        print(f"Data path: {eval_config.training_data}")
        
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        print("Evaluation completed successfully!")
        # evaluation.log_into_mlflow()


# This block should be at the module level, not indented inside the class
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e