import runpod
import os
from llm_api.llm_conf import API_KEY, ENDPOINT

class RunPodClient:
    def __init__(self, api_key, endpoint, timeout=60):
        runpod.api_key = api_key
        self.endpoint = runpod.Endpoint(endpoint)
        self.timeout = timeout

    def get_qwen_answer(self, prompt):
        run_request = self.endpoint.run_sync(
                {
                    "input": 
                    {
                        "prompt": prompt,

                        "sampling_params": 
                            {
                                "max_tokens": 256,
                                "temperature": 0.6,
                                "top_p": 0.6,
                                "repetition_penalty": 1.05

                            }                    
                    }
                },
                timeout=self.timeout,
            )
        
        try:
            # return run_request
            return run_request[0]['choices'][0]['tokens'][0]
        except:
            return "Ответ не был получен :("
        