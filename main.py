"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from typing import List

from telegram import (
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
import nltk
from gtts import gTTS

# Telegram bot token from https://t.me/BotFather
BOT_API_KEY = ""

# Path to file with fake logs and messages
FAKE_DATA_FILE = "fakes.json"

# Allowed characters
ALPHABET = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890- "

# Path to dialogues.txt
DIALOGUES_FILE = "dialogues.txt"

# Hyperparameter for dialogues.txt
DIALOGUES_THRESHOLD = 0.5

# Parsed fake messages and dialogues (global variables)
data = {}
dialogues_structured = {}


def fake_logs(logs: List[str]) -> None:
    """Prints fake logs in console

    Args:
        logs (List[str]): Log messages and delays after in seconds (ex.: ["Загрузка намереней из intents.json|0.1"])
    """
    for log in logs:
        log_message, delay = log.split("|")
        logging.info(log_message)
        time.sleep(float(delay))


def _parse_dialogues_from_file() -> None:
    """Loads and parses dialogues.txt
    Uses custom logs from fakes.json
    """
    global dialogues_structured
    # Загрузить данные диалогов из файла
    logging.warning(data["logs"]["dialogues_loading"].format(file_name=DIALOGUES_FILE))
    with open(DIALOGUES_FILE, "r", encoding="utf-8") as file:
        content = file.read()

    # Разделить по двойным строкам
    dialogues = [dialogue.split("\n")[:2] for dialogue in content.split("\n\n") if len(dialogue.split("\n")) == 2]

    # Отфильтровать повторяющиеся вопросы и отформатировать фразы
    logging.info(data["logs"]["dialogues_filtering"])
    dialogues_filtered = []
    questions = set()
    for dialogue in dialogues:
        question, answer = dialogue
        question = "".join(symbol for symbol in question[2:].lower() if symbol in ALPHABET).strip()
        answer = answer[2:]
        if question and question not in questions:
            questions.add(question)
            dialogues_filtered.append([question, answer])

    # Создать словарь слов и их связанных пар вопрос-ответ
    logging.info(data["logs"]["dialogues_structuring"])
    dialogues_structured = {}
    for question, answer in dialogues_filtered:
        words = set(question.split())
        for word in words:
            dialogues_structured.setdefault(word, []).append([question, answer])

    # Отсортировать пары по длине вопроса и оставить только первые 1000 для каждого слова
    logging.info(data["logs"]["dialogues_sort"])
    dialogues_structured = {
        word: sorted(pairs, key=lambda pair: len(pair[0]))[:1000] for word, pairs in dialogues_structured.items()
    }

    # Перемешать
    logging.info(data["logs"]["dialogues_shuffle"])
    dialogues_structured_list = list(dialogues_structured.items())
    random.shuffle(dialogues_structured_list)
    dialogues_structured = dict(dialogues_structured_list)

    # Готово
    logging.info(data["logs"]["dialogues_done"].format(words=len(dialogues_structured)))


def _generate_answer_dialogues(replica) -> str or None:
    """Returns best dialogue from dialogues.txt

    Args:
        replica (_type_): simplified request

    Returns:
        str or None: best response or None if not found
    """
    words = set(replica.split())
    mini_dataset = [pair for word in words if word in dialogues_structured for pair in dialogues_structured[word]]

    # [[distance_weighted, question, answer]]
    answers = []
    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < DIALOGUES_THRESHOLD:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < DIALOGUES_THRESHOLD:
                answers.append([distance_weighted, question, answer])

    return min(answers, key=lambda three: three[0])[2] if answers else None


async def _parse_and_send_fake_message(request: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Actual request processor

    Args:
        request (str): user's raw request
        update (Update): _description_
        context (ContextTypes.DEFAULT_TYPE): _description_
    """
    # User's ID
    chat_id = update.effective_chat.id

    # Simplify
    request = "".join(symbol for symbol in request.lower() if symbol in ALPHABET).strip()
    logging.info(data["logs"]["request_simplified"].format(request=request))

    # Try to find intent
    intent_name = None
    intent_content = {}
    for intent_name_, intent_content_ in data["intents"].items():
        request_regex = re.compile(intent_content_["request"])
        if request_regex.match(request):
            intent_name = intent_name_
            intent_content = intent_content_
    logging.info(data["logs"]["intent"].format(request=request, intent=intent_name))

    # Intent not found
    if not intent_name:
        logging.info(data["logs"]["intent_not_found"].format(intent=intent_name))

        # Try from dialogues.txt
        response = _generate_answer_dialogues(request)

        # Send it
        if response:
            await _send_safe(chat_id, response, context)

        # No response even in dialogues.txt -> send failure phrase
        else:
            await _send_safe(chat_id, random.choice(data["failures"]), context)

        return

    # Log topic if needed
    topic_current = intent_content.get("topic_current")
    topic_next = intent_content.get("topic_next")
    if topic_current:
        logging.info(
            data["logs"]["intent_topic"].format(
                user_id=chat_id,
                request=request,
                topic=topic_current,
                intent=intent_name,
            )
        )

    # Get response text
    responses = intent_content.get("responses")
    response = None
    if responses is not None:
        if isinstance(responses, list) and len(responses) != 0:
            response = random.choice(responses)
        elif isinstance(responses, str):
            response = responses

    # Send with voice
    if response and intent_content.get("voice"):
        await _send_safe(chat_id, response, context, voice=True)

    # Send as plain text or with image
    elif response:
        await _send_safe(chat_id, response, context, image=intent_content.get("image"))

    # Send sticker after message
    sticker_file_id = intent_content.get("sticker_file_id")
    if sticker_file_id:
        await _send_safe(chat_id, None, context, sticker_file_id=sticker_file_id)

    # Log next topic if needed
    if topic_next:
        logging.info(data["logs"]["topic_next"].format(topic=topic_next))


async def _send_safe(
    chat_id: int,
    text: str or None,
    context: ContextTypes.DEFAULT_TYPE,
    reply_to_message_id: int or None = None,
    markdown: bool = False,
    sticker_file_id: str or None = None,
    voice: bool = False,
    image: str or None = None,
):
    """Sends message without raising any error

    Args:
        chat_id (int): ID of user (or chat)
        text (str or None): text to send (can be None in case of sticker)
        context (ContextTypes.DEFAULT_TYPE): context object from bot's callback
        reply_to_message_id (int or None, optional): ID of message to reply on. Defaults to None
        markdown (bool, optional): True to parse as markdown. Defaults to False
        sticker_file_id (str or None, optional) file ID (usually starts with CAA...) to send as sticker
        voice (bool, optional): True to send as voice message. Defaults to False
        image (str or None, optional): path to image to send or None to send as text. Defaults to None
    """
    try:
        if sticker_file_id:
            await context.bot.send_sticker(chat_id=chat_id, sticker=sticker_file_id)
        elif voice:
            logging.info(data["logs"]["voice_generating"])
            gTTS(text=text, lang="ru").save("response.mp3")
            voice_msg = open("response.mp3", "rb")
            logging.info(data["logs"]["voice_sending"])
            await context.bot.send_voice(
                chat_id=chat_id,
                voice=voice_msg,
                caption=text,
                parse_mode="MarkdownV2" if markdown else None,
            )
            voice_msg.close()
            logging.info(data["logs"]["voice_deleting"])
            os.remove("response.mp3")
        elif image:
            with open(image, "rb") as file:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    caption=text,
                    photo=file,
                    reply_to_message_id=reply_to_message_id,
                    parse_mode="MarkdownV2" if markdown else None,
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=True,
                parse_mode="MarkdownV2" if markdown else None,
            )

    # Just ignore it
    except Exception:
        pass


async def _bot_callback_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _parse_and_send_fake_message("start", update, context)


async def _bot_callback_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _parse_and_send_fake_message(update.message.text.strip(), update, context)


def main() -> None:
    """Main entry"""
    global data

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Load fake data from JSON
    with open(FAKE_DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Load dialogues.txt
    _parse_dialogues_from_file()

    # Print initial logs
    fake_logs(data["logs"]["after_dialogues"])

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    builder = ApplicationBuilder().token(BOT_API_KEY)
    application = builder.build()
    application.add_handler(CommandHandler("start", _bot_callback_start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), _bot_callback_message))
    application.run_polling(close_loop=True)


if __name__ == "__main__":
    main()
