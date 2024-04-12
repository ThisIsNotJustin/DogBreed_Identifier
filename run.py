from fastapi import FastAPI
import gradio as gr

from DogBreedIdentifier import demo

app = FastAPI()

@app.get('/gradio')
async def root():
    return await app.run

app = gr.mount_gradio_appl(app, demo, path='/gradio')