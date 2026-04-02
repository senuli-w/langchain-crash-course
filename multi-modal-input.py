from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model('mistral-tiny', temperature=0.3)

message = {
    'role': 'user', 
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image_url', 'image_url': 'https://media.istockphoto.com/id/814423752/photo/eye-of-model-with-colorful-art-make-up-close-up.jpg?s=612x612&w=0&k=20&c=l15OdMWjgCKycMMShP8UK94ELVlEGvt7GmB_esHWPYE='}
    ]
}

response = model.invoke([message])

print(response.content)