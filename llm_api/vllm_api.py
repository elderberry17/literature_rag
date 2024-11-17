import runpod
import os

runpod.api_key = ""

endpoint = runpod.Endpoint("")

print(endpoint)

# run_request = endpoint.run_sync(
#         {
#             "input": {
#                 "prompt": "Ты умеешь говорить по-русски?",
#             }
#         },
#         timeout=60,
#     )



# можно вроде и через openai
# через эту либу часто с 433 падает
# print(run_request[0]['choices']['tokens'][0])

# в таком виде возвращает ответ
example = [{'choices': [{'tokens': [' Я умею!\nКонечно, я могу говорить на русском']}], 'usage': {'input': 11, 'output': 16}}]
print(example[0]['choices'][0]['tokens'][0])