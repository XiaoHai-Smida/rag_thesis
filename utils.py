from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from langchain_core.embeddings import Embeddings

"""
def get_embeddings(texts):
    model_id = "E:\myscripts\RAG_1\models"
    pipeline_se = pipeline(Tasks.sentence_embedding,
                        model=model_id,
                        sequence_length=512
                        , model_revision='master') # sequence_length 代表最大文本长度，默认值为128
    vectors = []
    for text in texts:
        inputs = {
                "source_sentence": [text]
            }
        result = pipeline_se(input=inputs)['text_embedding'][0]
        vectors.append(result.tolist())
    return vectors
"""

class get_embeddings(Embeddings):
    def __init__(self, model_path="/data/users/rwang/lib/models/BAAI/bge-m3"):
        self.model_id = model_path
        self.pipeline = pipeline(
            Tasks.sentence_embedding,
            model=self.model_id,
            sequence_length=512,
            model_revision='master'
        )
    
    def embed_documents(self, texts):  # 添加self参数
        vectors = []
        for text in texts:
            inputs = {
                    "source_sentence": [text]
                }
            result = self.pipeline(input=inputs)['text_embedding'][0]
            vectors.append(result.tolist())
        return vectors
    
    def embed_query(self, text):
        """Embed query text."""
        return self.embed_documents([text])[0]


if __name__=="__main__":
    query = ["国际争端"]
    documents = [
        "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
        "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
        "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
        "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
        "我国首次在空间站开展舱外辐射生物学暴露实验",
    ]
    a = get_embeddings(documents)
    print(len(a))
    for v in a:
        print(v[0:4])