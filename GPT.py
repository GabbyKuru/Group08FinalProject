import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import turtle
import openai
from openai import OpenAI

# Feeds in all the data into GPT, model can be swapped out.
# Tracks data on whether it guessed right or not.

human = 0
gpt = 0
human_right = 0
gpt_right = 0
total = 0

file_path = ""

client = OpenAI(api_key = '') # OPENAI KEY GOES THERE. it's linked to account and if it's uploaded
                              # then anyone with the key could cost me money. not doing that


def process_line(text, score):
    global human, gpt, human_right, gpt_right, total
    response = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09", # models are listed at https://platform.openai.com/docs/models/overview
        messages=[                      # I just used gpt-4-turbo-2024-04-09 and gpt-3.5-turbo-0125 cause gpt-4 is kind of expensive compared to those
            {"role": "system", "content": "Take a guess at whether the inputted essay is written by a human or written by ChatGPT. DO NOT give analysis or any words at all. ONLY output the number 0 if you think it's written by a human, or ONLY the number 1 if you think it's from ChatGPT. WRITE NO WORDS. ONLY THE NUMBER."},
            {"role": "user", "content": text}]
    )

    guess = response.choices[0].message.content
    print(guess)
    if score == 0:
        human += 1
        written = "human"
    elif score == 1:
        gpt += 1
        written = "gpt"
    total = human + gpt

    text_guess = ""    # Weird formatting here cause it kept putting 0 or 0. even after saying don't do that
    if (guess == "0" or guess == "0."): text_guess = "human"
    if (guess == "1" or guess == "1."): text_guess = "gpt"

    if (guess == "0" or guess == "0.") and score == 0: human_right += 1
    elif (guess == "1" or guess == "1.") and score == 1: gpt_right += 1


with open(file_path, encoding="utf8") as file:
    csv_reader = csv.reader(file)
    first = total
    for _ in range(1):
        next(csv_reader)
    for row in csv_reader:
        text = row[0]
        score = int(row[1])
        process_line(text, score)
        print("Human Right: " + str(human_right) + " out of " + str(human))
        print("GPT Right: " + str(gpt_right) + " out of " + str(gpt))
        if first+30 == total:
            first = total
            time.sleep(15)  # so it doesn't go over the api's requests/min limit