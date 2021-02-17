from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

text_kr = ""
with open("basic/tts/input.txt", "r", encoding='utf-8') as f:
    for line in f:
        text_kr += line.replace(".", ". ")

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text=text_kr)

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="ko-KR", 
    ssml_gender=texttospeech.SsmlVoiceGender.MALE
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=0.75
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, 
    voice=voice, 
    audio_config=audio_config
)

# The response's audio_content is binary.
with open("genesis.mp3", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "genesis.mp3"')
