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

column references
- เกี่ยวกับจังหวัด: ในคอลัมน์ 'BUDGETARY_UNIT'

จงทำการค้นหาข้อมูลเกี่ยวกับ 
{instruction_str}

คำถาม: {query_str}

คำตอบ: """
)

instruction_str = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""


df = pd.read_csv("[67] Thailand's Budget - ขาวคาดแดง RELEASE_2023-12-30.csv")
df["AMOUNT"] = df["AMOUNT"].str.replace(",", "").astype(float)
df = df.fillna(df.dtypes.replace({"object": "", "int64": 0, "float64": 0}))

query_engine = PandasQueryEngine(df=df, verbose=True)
query_engine.update_prompts({"pandas_prompt": new_prompt, "instruction_str": instruction_str})


def query_pipeline(query_str):
    query_str = query_str.replace("ทั้งหมดเท่าไหร่", "")
    query_str = query_str.replace("งบ", "")
    query_str = query_str.replace("งบประมาณ", "")
    response = query_engine.query(query_str)
    return response


@app.get("/query")
def query(query: str):
    response = query_engine.query(query)
    return {"response": response.response}


@app.get("/bento")
def bento(query: str):
    response = query_pipeline(query)
    results_df_query = eval(response.metadata["pandas_instruction_str"])

    return {
        "categories": results_df_query.groupby("CATEGORY_LV1")["AMOUNT"].sum().to_dict(),
        "total": results_df_query["AMOUNT"].sum(),
        "total_projects": len(results_df_query),
    }
