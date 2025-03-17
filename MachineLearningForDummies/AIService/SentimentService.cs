using MachineLearningForDummies.AIModel;

namespace MachineLearningForDummies.AIService
{
    public class SentimentService
    {
        private readonly ModelPredictor _predictor;

        public SentimentService()
        {
            _predictor = new ModelPredictor();
        }

        public ModelOutput PredictSentiment(string text)
        {
            return _predictor.Predict(text);
        }

        public void AddTrainingData(string text, bool label)
        {
            ModelTrainer.AppendTrainingData(text, label);
            ModelTrainer.TrainAndSaveModel();
            _predictor.ReloadModel();
        }
    }
}
