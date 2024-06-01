import pyttsx3

 #making this thorat

engine=pyttsx3.init("sapi5") #microsoft speaking software
voices = engine.getProperty('voices')
engine.setProperty('voices',voices[0].id)
engine.setProperty('rate',170) # speed of speech

def Say(Text):
    print("   ")
    print(f"A.I':{Text}")
    engine.say(text = Text)
    engine.runAndWait()
    print("  ")
    
          
