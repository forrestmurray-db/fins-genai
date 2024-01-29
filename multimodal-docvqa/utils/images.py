import base64

from openai import OpenAI

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def gpt_summarize_image(client, img, prompt=None, caption=None):
  base64_image = encode_image(img.metadata.image_path)

  prompt = prompt or "Describe the image in detail. Be specific about graphs, such as bar plots."

  if caption:
    prompt = prompt + "A caption for the image {caption}"

  client = OpenAI(api_key=dbutils.secrets.get("access-tokens", 'openai'))

  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ],
      }
    ],
    max_tokens=300,
  )

  return response.choice[0]