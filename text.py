import paddle
from paddlespeech.cli.text import TextExecutor
text_executor = TextExecutor()



result = text_executor(text='今天的天气真不错啊你下午有空吗我想约你一起去吃饭',task='punc',model='ernie_linear_p7_wudao',lang='zh',config=None,ckpt_path=None,punc_vocab=None,device=paddle.get_device())

result = text_executor(text='对此工信部副部长辛国斌表示新能源汽车是全球汽车产业转型升级和绿色发展的主要方向也是我国汽车产业高质量发展的战略选择70年来我国汽车产业从无到有从小到大从弱到强第2000万辆新能源汽车的下线是一个具有历史意义的重要时刻',task='punc',model='ernie_linear_p7_wudao',lang='zh',config=None,ckpt_path=None,punc_vocab=None,device=paddle.get_device())


print('Text Result: \n{}'.format(result))
