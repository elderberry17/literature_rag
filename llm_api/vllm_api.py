import runpod
import os

# создаем в начале
# runpod.api_key = ""
# endpoint = runpod.Endpoint("")

class RunPodClient:
    def __ini__(self, api_key, endpoint, timeout=60):
        runpod.api_key = api_key
        self.endpoint = runpod.Endpoint(endpoint)
        self.timeout = timeout

    def get_qwen_answer(self, prompt):
        run_request = self.endpoint.run_sync(
                {
                    "input": {
                        "prompt": prompt,
                    }
                },
                timeout=self.timeout,
            )
        
        try:
            return run_request[0]['choices'][0]['tokens'][0]
        except:
            return "Ответ не получен :("
        
