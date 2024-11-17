from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from llm_api.llm_conf import API_KEY, ENDPOINT
from llm_api.vllm_api import RunPodClient
from retrieval.qdrant_conf import COLLECTION_NAME, VECTORIZER_PATH, QDRANT_URL
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src_bot.main_conf import TELEGRAM_BOT_TOKEN, TOKENIZER_NAME
from src_bot.prompts import SYSTEM_PROMPT, make_prompt
from transformers import AutoTokenizer
from qdrant_client import QdrantClient


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
    react to /start
    '''
    await update.message.reply_text("Я помогу тебе ответить на тест по Капитанской Дочке. Задавай свои вопросы и предлагай варианты ответа.")


async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
    react to /ask
    '''
    question = ' '.join(context.args)  
    if not question:
        await update.message.reply_text("Пожалуйста, задайте вопрос.")
        return

    prompt = make_prompt(question, QDRANT_CLIENT, TOKENIZER, VECTORIZER, COLLECTION_NAME, SYSTEM_PROMPT)
    print(f"PROMPT:\n{prompt}")
    response = RUNPOD_CLIENT.get_qwen_answer(prompt)

    await update.message.reply_text(response)


if __name__ == "__main__":
    # load all the necessary things
    RUNPOD_CLIENT = RunPodClient(API_KEY, ENDPOINT)
    with open(VECTORIZER_PATH, "rb") as f:
        VECTORIZER = pickle.load(f)
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    QDRANT_CLIENT = QdrantClient(url=QDRANT_URL)

    # tg bot
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask_question))
    
    print("Bot is running...")
    app.run_polling()
