import pyttsx3 as tts

engine = tts.init()

def say(text, speed=150, voice=1):
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', speed)
    engine.say(text)
    engine.runAndWait()