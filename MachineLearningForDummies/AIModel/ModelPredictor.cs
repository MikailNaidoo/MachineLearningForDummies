using Microsoft.ML;

namespace MachineLearningForDummies.AIModel
{
    public class ModelPredictor
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<ModelInput, ModelOutput> _predictionEngine;
        private string _modelPath;

        public ModelPredictor()
        {
            _mlContext = new MLContext();
            _modelPath = Path.Combine(Directory.GetCurrentDirectory(), "AIModel", "Model.zip");
            LoadModel();
        }

        private void LoadModel()
        {
            if (File.Exists(_modelPath))
            {
                var trainedModel = _mlContext.Model.Load(_modelPath, out _);
                _predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            }
            else
            {
                Console.WriteLine("No trained model found. Please train the model first.");
            }
        }

        public ModelOutput Predict(string inputText)
        {
            if (_predictionEngine == null)
            {
                return new ModelOutput { Prediction = false, Probability = 0.0f };
            }

            var input = new ModelInput { Text = inputText };
            return _predictionEngine.Predict(input);
        }

        public void ReloadModel()
        {
            LoadModel();
        }
    }
}
