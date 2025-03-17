using Microsoft.ML;
using System.Text.Json;

namespace MachineLearningForDummies.AIModel
{
    public class ModelTrainer
    {
        private static readonly string DataPath = Path.Combine(Directory.GetCurrentDirectory(), "training-data.json");
        private static readonly string ModelPath = Path.Combine(Directory.GetCurrentDirectory(), "AIModel", "Model.zip");
        private static readonly object _lock = new(); // Thread safety

        public static void TrainAndSaveModel()
        {
            lock (_lock) // Prevent concurrent writes
            {
                var mlContext = new MLContext();

                // Load training data from JSON
                List<ModelInput> trainingData = LoadTrainingData();

                // Convert to IDataView
                IDataView data = mlContext.Data.LoadFromEnumerable(trainingData);

                // Define training pipeline
                var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text").Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

                // Train the model
                var model = pipeline.Fit(data);

                // Save the trained model
                mlContext.Model.Save(model, data.Schema, ModelPath);

                Console.WriteLine($"Model retrained and saved to {ModelPath}");
            }
        }

        public static List<ModelInput> LoadTrainingData()
        {
            if (!File.Exists(DataPath))
            {
                return new List<ModelInput>();
            }

            string json = File.ReadAllText(DataPath);
            return JsonSerializer.Deserialize<List<ModelInput>>(json) ?? new List<ModelInput>();
        }

        public static void AppendTrainingData(string text, bool label)
        {
            lock (_lock)
            {
                List<ModelInput> trainingData = LoadTrainingData();
                trainingData.Add(new ModelInput { Text = text, Label = label });

                // Save back to JSON
                string updatedJson = JsonSerializer.Serialize(trainingData, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(DataPath, updatedJson);
            }
        }
    }
}
