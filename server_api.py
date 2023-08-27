import uvicorn
from fastapi import FastAPI
import paddle
from paddlespeech.cli.text import TextExecutor

app = FastAPI()
text_executor = TextExecutor()
text_executor._init_from_path()
paddle_device = paddle.get_device()

@app.get("/punctuation")
def get_punctuation(text: str):
    print(text)
    punctuation_res = text_executor(text=text,task='punc',model='ernie_linear_p7_wudao',lang='zh',
                                    config=None,ckpt_path=None,punc_vocab=None,device=paddle_device)
    return {"result": punctuation_res}

if __name__ == "__main__":
    uvicorn.run(app="server_api:app", host='0.0.0.0', port=50102, reload=False)
