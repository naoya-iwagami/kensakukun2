import os  
import json  
import threading  
import datetime  
import uuid  
import traceback  
import base64  
  
from flask import Flask, request, render_template, session, send_file  
from flask_session import Session  
  
from azure.search.documents import SearchClient  
from azure.core.credentials import AzureKeyCredential  
from azure.core.pipeline.transport import RequestsTransport  
from azure.cosmos import CosmosClient  
import openai  
import certifi  
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  
  
from markdown2 import markdown  
import io  
from urllib.parse import quote, unquote  
from concurrent.futures import ThreadPoolExecutor  
  
# Flask/Session初期化  
app = Flask(__name__)  
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key')  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  
app.config['SESSION_PERMANENT'] = False  
Session(app)  
  
# Azure初期化  
client = openai.AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview"),  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  
)  
  
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
search_service_key = os.getenv("AZURE_SEARCH_KEY")  
transport = RequestsTransport(verify=certifi.where())  
  
cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
cosmos_key = os.getenv("AZURE_COSMOS_KEY")  
database_name = 'chatdb'  
container_name = 'kensakukun2'  
cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)  
database = cosmos_client.get_database_client(database_name)  
container = database.get_container_client(container_name)  
  
blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)  
main_container_name = 'filetest11'  
lock = threading.Lock()  
  
# ベクトル検索用定数  
EMBEDDING_MODEL_SMALL = "text-embedding-3-small"  
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"  
EMBEDDING_DIM = 1536  
INDEX_CURRENT_SMALL = "filetest11"  
INDEX_NEW_SMALL = "index11-small"  
INDEX_LARGE = "filetest11-large"  
  
def extract_account_key(connection_string):  
    pairs = [s.split("=", 1) for s in connection_string.split(";") if "=" in s]  
    conn_dict = dict(pairs)  
    return conn_dict.get("AccountKey")  
  
def generate_sas_url(blob_client, blob_name):  
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
    account_key = extract_account_key(connection_string)  
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)  
    expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)  
    sas_token = generate_blob_sas(  
        account_name=blob_client.account_name,  
        container_name=blob_client.container_name,  
        blob_name=blob_name,  
        account_key=account_key,  
        permission=BlobSasPermissions(read=True),  
        expiry=expiry,  
        start=start  
    )  
    url = f"{blob_client.url}?{sas_token}"  
    return url  
  
def get_authenticated_user():  
    # --- Easy Auth (EntraID)方式でIDを取る ---  
    # セッションから取得（キャッシュ済ならそれを使う）  
    if "user_id" in session and "user_name" in session:  
        return session["user_id"]  
    client_principal = request.headers.get("X-MS-CLIENT-PRINCIPAL")  
    if client_principal:  
        try:  
            decoded = base64.b64decode(client_principal).decode("utf-8")  
            user_data = json.loads(decoded)  
            user_id = None  
            user_name = None  
            if "claims" in user_data:  
                for claim in user_data["claims"]:  
                    if claim.get("typ") == "http://schemas.microsoft.com/identity/claims/objectidentifier":  
                        user_id = claim.get("val")  
                    if claim.get("typ") == "name":  
                        user_name = claim.get("val")  
            if user_id:  
                session["user_id"] = user_id  
            if user_name:  
                session["user_name"] = user_name  
            return user_id  
        except Exception as e:  
            print("Easy Auth ユーザー情報の取得エラー:", e)  
    session["user_id"] = "anonymous"  
    session["user_name"] = "anonymous"  
    return session["user_id"]  
  
def get_search_client(index_name):  
    return SearchClient(  
        endpoint=search_service_endpoint,  
        index_name=index_name,  
        credential=AzureKeyCredential(search_service_key),  
        transport=transport  
    )  
  
def get_query_embedding(query, model_name, embed_dim):  
    response = client.embeddings.create(  
        input=query,  
        model=model_name,  
        dimensions=embed_dim  
    )  
    emb = response.data[0].embedding  
    assert len(emb) == embed_dim, f"Embeddingサイズ不一致: 実際 {len(emb)}, 指定 {embed_dim}"  
    return emb  
  
def vector_search(query, topNDocuments, index_name, embedding_model, embed_dim):  
    query_embedding = get_query_embedding(query, embedding_model, embed_dim)  
    vector_query = {  
        "kind": "vector",  
        "vector": query_embedding,  
        "exhaustive": True,  
        "fields": "contentVector",  
        "weight": 0.5,  
        "k": topNDocuments  
    }  
    search_client = get_search_client(index_name)  
    results = search_client.search(  
        search_text="*",  
        vector_queries=[vector_query],  
        select="title, content, filepath"  
    )  
    results_list = list(results)  
    if results_list and "@search.score" in results_list[0]:  
        results_list.sort(key=lambda x: x.get("@search.score", 0), reverse=True)  
    return results_list  
  
def prepare_files(results):  
    files = []  
    for result in results:  
        filepath = result.get('filepath', '')  
        title = result.get('title', '不明')  
        content = result.get('content', '')  
        if filepath.lower().endswith('.txt'):  
            url = f"/download_txt/{main_container_name}/{quote(filepath)}"  
        elif filepath:  
            blob_client = blob_service_client.get_blob_client(container=main_container_name, blob=filepath)  
            url = generate_sas_url(blob_client, filepath)  
        else:  
            url = ''  
        files.append({'title': title, 'content': content, 'url': url})  
    return files  
  
def make_context_for_assistant(results):  
    return "\n---\n".join([  
        f"タイトル: {doc.get('title', '不明')}\n内容: {doc.get('content', '')[:500]}"  
        for doc in results  
    ])  
  
def assistant_answer(prompt, context):  
    system_prompt = (  
        "あなたは親切な日本語AIアシスタントです。"  
        "与えられたコンテキスト資料と質問から、迅速・正確に答えてください。"  
        "分からない場合は無理に創作せず、「分かりません」と返してください。"  
    )  
    messages = [  
        {"role": "system", "content": system_prompt},  
        {"role": "user", "content": f"質問: {prompt}\n\n資料:\n{context}"}  
    ]  
    answer = client.chat.completions.create(  
        model="gpt-4.1",  
        messages=messages,  
        max_tokens=1024,  
        temperature=0,  
    ).choices[0].message.content  
    return markdown(answer, extras=["tables", "break-on-newline"])  
  
@app.route('/')  
def index():  
    return render_template('index.html')  
  
@app.route('/send_message', methods=['POST'])  
def send_message():  
    data = request.get_json()  
    prompt = data.get('prompt', '').strip()  
    if not prompt:  
        return json.dumps({'error': 'Empty input'}), 400, {'Content-Type': 'application/json'}  
  
    topN = 15  
    try:  
        # 検索およびLLM応答を並列で実行  
        executor = ThreadPoolExecutor(max_workers=6)  
        future_vs = {  
            'current_small': executor.submit(  
                vector_search, prompt, topN, INDEX_CURRENT_SMALL, EMBEDDING_MODEL_SMALL, EMBEDDING_DIM  
            ),  
            'new_small': executor.submit(  
                vector_search, prompt, topN, INDEX_NEW_SMALL, EMBEDDING_MODEL_SMALL, EMBEDDING_DIM  
            ),  
            'large': executor.submit(  
                vector_search, prompt, topN, INDEX_LARGE, EMBEDDING_MODEL_LARGE, EMBEDDING_DIM  
            )  
        }  
        results_vs = {k: f.result() for k, f in future_vs.items()}  
        future_llm = {  
            k: executor.submit(  
                assistant_answer, prompt, make_context_for_assistant(v)  
            )  
            for k, v in results_vs.items()  
        }  
        answers = {k: f.result() for k, f in future_llm.items()}  
        resp = {  
            'current_small_answer': answers['current_small'],  
            'new_small_answer': answers['new_small'],  
            'large_answer': answers['large'],  
            'current_small_files': prepare_files(results_vs['current_small']),  
            'new_small_files': prepare_files(results_vs['new_small']),  
            'large_files': prepare_files(results_vs['large'])  
        }  
        # (ユーザーID取得は後続APIで利用するためメモ)  
        _ = get_authenticated_user()  
        return json.dumps(resp, ensure_ascii=False), 200, {'Content-Type': 'application/json'}  
    except Exception as e:  
        print("ベクトル検索アシスタントエラー:", e)  
        traceback.print_exc()  
        return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}  
  
@app.route("/download_txt/<container>/<path:blobname>")  
def download_txt(container, blobname):  
    blobname = unquote(blobname)  
    blob_client = blob_service_client.get_blob_client(container=container, blob=blobname)  
    txt_bytes = blob_client.download_blob().readall()  
    try:  
        txt_str = txt_bytes.decode("utf-8")  
    except UnicodeDecodeError:  
        txt_str = txt_bytes.decode("cp932", errors="ignore")  
    bom = b'\xef\xbb\xbf'  
    buf = io.BytesIO(bom + txt_str.encode("utf-8"))  
    filename = os.path.basename(blobname)  
    ascii_filename = "download.txt"  
    response = send_file(  
        buf,  
        as_attachment=True,  
        download_name=ascii_filename,  
        mimetype="text/plain; charset=utf-8"  
    )  
    response.headers["Content-Disposition"] = (  
        f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{quote(filename)}'  
    )  
    return response  
  
@app.route('/save_rating', methods=['POST'])  
def save_rating():  
    try:  
        data = request.get_json()  
        user_id = get_authenticated_user()  
        user_prompt = data.get("user_prompt", "")  
        ratings = data.get("ratings", {})       # dict: column名→score  
        assistant_answers = data.get("assistant_answers", {}) # dict: column名→content  
  
        # 入力チェック  
        for col in ["current_small", "new_small", "large"]:  
            if not str(ratings.get(col, "")) or not str(assistant_answers.get(col, "")):  
                return json.dumps({"result": "error", "error": f"{col} の評価または回答が不足"}), 400, {"Content-Type":"application/json"}  
  
        item = {  
            "id": str(uuid.uuid4()),  
            "user_id": user_id,  
            "user_prompt": user_prompt,  
            "ratings": ratings,  
            "assistant_answers": assistant_answers,  
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()  
        }  
        with lock:  
            container.upsert_item(item)  
        return json.dumps({"result": "ok"}), 200, {"Content-Type":"application/json"}  
    except Exception as e:  
        print("評価保存失敗:", e)  
        return json.dumps({"result": "error", "error": str(e)}), 500, {"Content-Type":"application/json"}  
  
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0')  