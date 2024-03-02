# Author: Michael Suliot (Michael AI)
# Date: 8/5/2023 - 1.0
# - Update 3/2/2024 - 1.1 - added misssing requirement and retested
# Version: Beta 1.1
# Description: Quick sample converting text to speech using Huggingface using local model
# Project: huggingface_text_to_speech

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import nltk
from nltk.tokenize import word_tokenize


def hf_local(text):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=text, return_tensors="pt")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)


def main():
    text = "Hello, my name is Michael, and I want to thank you for watching my videos on getting started with Hugging Face"

    def count_tokens(text):
        tokens = word_tokenize(text)
        return len(tokens)

    def count_words(text):
        text = text.strip()
        words = text.split()
        return len(words)

    # token_count = count_tokens(text)
    # word_count = count_words(text)
    total_length = len(text)
    print("token count:",count_tokens(text))
    print("word count:",count_words(text))
    print("total length:", len(text))

    if total_length > 600   :
        print(f"Text is too long. Total length is {total_length} characters. Please reduce to 550 characters or less.")
    else:
        hf_local(text)
        print("Text to speech conversion complete. Please check the speech.wav file in your current directory.")

if __name__ == "__main__":
    main()