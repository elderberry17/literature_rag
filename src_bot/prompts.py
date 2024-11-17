from retrieval.retrieval_qdrant import find_chunks


ANNOTATION = """В романе «Капитанская дочка» А.С.Пушкин нарисовал яркую картину стихийного крестьянского восстания под предводительством Емельяна Пугачева.
В романе 14 глав
Береги честь смолоду.
Пословица"""

SYSTEM_PROMPT = f"""Ты работаешь литературным экспертом по роману "Капитанская дочка".
Твоя задача - фактологически верно отвечать на вопросы о романе.
Старайся отвечать коротко и по делу.
Если ты предлагаешь несколько вариантов, выбери один самый релевантный.
Отвечай только на русском языке и не используй другие.
Если в вопросе приведены варианты ответа, ты можешь выбрать один из них.
Аннотация произведения: {ANNOTATION}"""

def make_prompt(user_content, qdrant_client, tokenizer, vectorizer, collection_name, system_prompt, limit=1):
    user_content_parts = user_content.split('ВАРИАНТЫ ОТВЕТА')
    user_content = user_content_parts[0]
    options = user_content_parts[1] if len(user_content_parts) > 1 else ""

    relevant_chunks = find_chunks(user_content, vectorizer, collection_name, qdrant_client, limit=limit)
    context_chunks = '\n'.join(relevant_chunks)
    user_content = 'Вопрос:\n' + user_content + '\nКонтекст:\n' + context_chunks + '\nВарианты ответа:\n' + options
    
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content}
    ]

    prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    return prompt
