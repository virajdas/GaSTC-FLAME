import sys
import threading
import tkinter as tk

import speech_recognition as sr
import pyttsx3 as tts

from stablediffusion import stable_diffusion
from music2 import music

import random

import wikipedia as wiki

class Flame:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.speaker = tts.init()
        self.speaker.setProperty("rate", 150)
        self.voices = self.speaker.getProperty('voices')
        self.speaker.setProperty('voice', self.voices[1].id)

        self.root = tk.Tk()
        self.root.title("Flame - FineArts & Language AI Merged Engine")
        self.root.geometry("1372x772")

        self.img1 = tk.PhotoImage(file="Idle 1x.png")
        self.img2 = tk.PhotoImage(file="Active 1x.png")

        self.canvas = tk.Canvas(self.root, bg="#0a0526", height=772, width=1372, bd=0, highlightthickness=0, relief="ridge")
        self.canvas.place(x=0, y=0)

        self.image_container = self.canvas.create_image(0, 0, anchor="nw", image=self.img1)

        threading.Thread(target=self.run_assistant).start()

        self.root.resizable(False, False)
        self.root.mainloop()

    def run_assistant(self):
        while True:
            try:
                with sr.Microphone() as mic:
                    self.recognizer.adjust_for_ambient_noise(mic)
                    audio = self.recognizer.listen(mic)
                    text = self.recognizer.recognize_google(audio)
                    text = text.lower()
                    print(text)

                    if "flame" in text or "slim" in text or "plane" in text:
                        self.canvas.itemconfig(self.image_container, image=self.img2)
                        audio = self.recognizer.listen(mic)
                        text = self.recognizer.recognize_google(audio)
                        text = text.lower()
                        print(text)
                        if "exit" in text or "bye" in text:
                            self.speaker.say(random.choice(["Bye!", "See you!", "Goodbye!", "Till next time!"]))
                            self.speaker.runAndWait()
                            self.speaker.stop()
                            self.root.destroy()
                            sys.exit()
                        else:
                            if text is not None:
                                if "art" in text and "mozart" not in text:
                                    self.speaker.say("What shall I make?")
                                    self.speaker.runAndWait()
                                    audio = self.recognizer.listen(mic)
                                    text = self.recognizer.recognize_google(audio)
                                    text = text.lower()
                                    self.speaker.say("Then so it shall be! I will begin creating!")
                                    self.speaker.runAndWait()
                                    stable_diffusion(text)
                                    self.speaker.say("Done! It is now saved under the Exported Art folder.")
                                    self.speaker.runAndWait()
                                elif "music" in text:
                                    self.speaker.say("Ah yes music, I love music. Just give me a few moments to compile the notes.")
                                    self.speaker.runAndWait()
                                    music()
                                    self.speaker.say(f"Done! Tell me, am I the next {random.choice(['Mozart', 'Beethoven'])}? Actually, don't tell me, I already know I am.")
                                    self.speaker.runAndWait()
                                elif "search" in text or "mozart" in text:
                                    replaced = text.replace('search', '')
                                    search_res = wiki.summary(replaced, 1)
                                    self.speaker.say(f"Here's what I found on the web: {search_res}")
                                    self.speaker.runAndWait()
                                else:
                                    self.speaker.say("I am so sorry, I didn't quite catch that. Could you please repeat?")
                                    self.speaker.runAndWait()
                            self.canvas.itemconfig(self.image_container, image=self.img1)
            except:
                self.canvas.itemconfig(self.image_container, image=self.img1)
                continue

if __name__ == '__main__':
    Flame()