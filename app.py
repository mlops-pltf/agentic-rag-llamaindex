from dotenv import load_dotenv
from os.path import join, dirname
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

from flask import Flask, request
from rag_utilities.db import get_allowed_data_genres
from rag_utilities.agent import invoke_rag_agent_workflow, upload_data



app = Flask(__name__)


# Run the system
@app.route("/ask_ques", methods=["POST"])
async def ask_questions():
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = await invoke_rag_agent_workflow(user_msg=query)
    print(response)
    response_answer = {
        "answer": str(response)
    }
    return response_answer

@app.route("/upload_doc", methods=["POST"])
async def doc_uploader():
    data_genre = request.form.getlist('data_genre')[0]
    if data_genre.lower() not in get_allowed_data_genres():
        raise NotImplementedError(f"Data Genre - '{data_genre}' is not implemented yet. Allowed data genres are - {get_allowed_data_genres()}")
    print(f"Uploading {data_genre} data...")
    file = request.files["file"]
    file_name = file.filename
    save_file = f"{data_genre}_data/" + file_name
    print(f"Saving file on device...")
    file.save(save_file)
    print(f"Uploading file - {file_name} to vector DB...")
    await upload_data(data_genre, [save_file])
    print("Successfully Uploaded.")
    response = {
        "status": "Successfully Uploaded"
        , "filename": file_name
    }
    return response

if __name__ == '__main__':
    app.run(
        host='0.0.0.0'
        , port=8080
        , debug=True
    )