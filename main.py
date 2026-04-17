from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os

app = FastAPI()

# Web版（Flutter Web）から通信できるようにするための設定（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 本番ではVercelのURLにするのが安全ですが、まずは全許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini APIの設定
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 推奨モデルの設定
model = genai.GenerativeModel('gemini-3-flash-preview') # 最新の高速モデル

@app.get("/")
def read_root():
    return {"message": "Study App Backend is running!"}

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")

    try:
        # 画像の読み込み
        image_data = await file.read()
        image_parts = [
            {
                "mime_type": file.content_type,
                "data": image_data
            }
        ]

        # Geminiへの厳密なプロンプト（仕様書通り）
        prompt = """
        あなたはプロの塾講師です。以下の画像には小テストの問題が含まれています。
        以下のルールに厳密に従い、JSON形式のみで出力してください。Markdownのコードブロック(```json)などは一切不要です。

        # ■ 問題分割ルール
        - 画像からテキストを抽出し、問題を「大問・小問構造」に分割すること。
        - 小問単位で1つの問題とする。
        - 小問が属する大問の本文は必ず保持すること。
        - 問題番号は画像の表記を尊重すること（例：(1)、(ア)など）。
        - question_idは「大問番号-小問番号」の形式で生成すること（例: 2-1, 3-ア）。
        - 各問題の解答と詳細な解説を作成すること。

        # ■ 出力形式
        以下のJSON形式のみを出力すること。余計な文章は一切出力しないこと。
        {
          "questions": [
            {
              "question_id": "string",
              "main_question": "string",
              "sub_question": "string",
              "answer": "string",
              "explanation": "string"
            }
          ]
        }
        """

        # Gemini APIを呼び出し
        response = model.generate_content([prompt, image_parts[0]])
        
        # Geminiの返答からJSON部分だけを抜き出す安全処理
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        
        # JSONとして解析して返す
        return json.loads(response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
