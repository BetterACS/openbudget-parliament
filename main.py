import logging
import sys

import dotenv
import fastapi
import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine import PandasQueryEngine

dotenv.load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from fastapi import FastAPI

app = FastAPI()


new_prompt = PromptTemplate(
    """\
คุณกำลังทำงานเกี่ยวกับ งบประมาณของประเทศไทย
และนี่คือข้อมูลคล่าวๆ ของข้อมูลที่คุณกำลังใช้งาน
{df_str}

จงทำการค้นหาข้อมูลเกี่ยวกับ 
{instruction_str}
คำถาม: {query_str}

คำตอบ: """
)

df = pd.read_csv("[67] Thailand's Budget - ขาวคาดแดง RELEASE_2023-12-30.csv")
query_engine = PandasQueryEngine(df=df, verbose=True)
query_engine.update_prompts({"pandas_prompt": new_prompt})


@app.get("/query")
def query(query: str):
    response = query_engine.query(query)
    return {"response": response.response}
