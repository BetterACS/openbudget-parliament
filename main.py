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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

new_prompt = PromptTemplate(
    """\
คุณกำลังทำงานเกี่ยวกับ งบประมาณของประเทศไทย
และนี่คือข้อมูลคล่าวๆ ของข้อมูลที่คุณกำลังใช้งาน
{df_str}

column references
- เกี่ยวกับจังหวัด: ในคอลัมน์ 'BUDGETARY_UNIT'

จงทำการค้นหาข้อมูลเกี่ยวกับ 
{instruction_str}

หากเกี่ยวข้องกับจังหวัดกรุณาเติมคำว่า 'จังหวัด' ตามด้วยชื่อจังหวัดในการค้นหา
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

with open("jungwat.txt", "r", encoding="utf-8") as f:
    jungwats = f.readlines()
jungwats = [j.strip() for j in jungwats]

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

    if query.replace("จังหวัด", "").replace("เมือง", "") == "พัทยา":
        results_df_query = df[df["BUDGETARY_UNIT"].str.contains("พัทยา")]

    # results_df_query = results_df_query[results_df_query["BUDGETARY_UNIT"].str.contains(query.replace("จังหวัด", ""))]
    # if query.replace("จังหวัด", "") in ["กรุงเทพ", "กรุงเทพมหานคร", "กทม", "เมืองพัทยา", "พัทยา"]:
    #     results_df_query = results_df_query[results_df_query["MINISTRY"] == "องค์กรปกครองส่วนท้องถิ่น"]
    # el
    if query.replace("จังหวัด", "") in jungwats:
        results_df_query = results_df_query[
            (results_df_query["MINISTRY"] == "จังหวัดและกลุ่มจังหวัด")
            | (results_df_query["MINISTRY"] == "องค์กรปกครองส่วนท้องถิ่น")
        ]
        # return str(results_df_query)

    return {
        "categories": [(k, v) for k, v in results_df_query.groupby("CATEGORY_LV1")["AMOUNT"].sum().to_dict().items()],
        "strategies": [(k, v) for k, v in results_df_query.groupby("STRATEGY")["AMOUNT"].count().to_dict().items()],
        "ministry": [(k, v) for k, v in results_df_query.groupby("MINISTRY")["AMOUNT"].count().to_dict().items()],
        "total": results_df_query["AMOUNT"].sum(),
        "total_projects": len(results_df_query),
    }
