using MachineLearningForDummies.AIModel;
using MachineLearningForDummies.AIService;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MachineLearningForDummies.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SentimentController : ControllerBase
    {
        private readonly SentimentService _sentimentService;

        public SentimentController()
        {
            _sentimentService = new SentimentService();
        }

        [HttpPost("predict")]
        public IActionResult Predict([FromBody] string text)
        {
            var result = _sentimentService.PredictSentiment(text);
            return Ok(result);
        }

        [HttpPost("train")]
        public IActionResult Train([FromBody] ModelInput data)
        {
            _sentimentService.AddTrainingData(data.Text, data.Label);
            return Ok("Model retrained successfully.");
        }
    }
}
